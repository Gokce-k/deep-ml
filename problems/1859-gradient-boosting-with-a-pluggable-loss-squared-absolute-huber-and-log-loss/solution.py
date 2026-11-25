import numpy as np


def _gbm_sigmoid(scores):
    return np.exp(-np.logaddexp(0.0, -scores))


def _gbm_tree_predict(tree, X):
    predictions = np.empty(X.shape[0], dtype=float)

    for i, row in enumerate(X):
        node = tree

        while "value" not in node:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]

        predictions[i] = node["value"]

    return predictions


def gbm_fit(X, y, loss, M, nu, max_depth, delta=1.0):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2 or len(y) == 0 or X.shape[0] != len(y):
        raise ValueError("X ve y boyutlari uyusmali.")

    if loss not in ("squared", "absolute", "huber", "logloss"):
        raise ValueError("Gecersiz loss.")

    if loss == "huber" and delta <= 0:
        raise ValueError("delta pozitif olmali.")

    # Başlangıçtaki sabit tahmin
    if loss == "squared":
        f0 = float(np.mean(y))

    elif loss in ("absolute", "huber"):
        f0 = float(np.median(y))

    else:
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Log loss etiketleri 0 veya 1 olmali.")

        rate = float(np.clip(np.mean(y), 1e-12, 1.0 - 1e-12))
        f0 = float(np.log(rate / (1.0 - rate)))

    model = {
        "f0": f0,
        "nu": float(nu),
        "n_features": X.shape[1],
        "trees": [],
    }

    f = np.full(len(y), f0, dtype=float)
    losses = []

    def sse(values):
        return float(np.sum((values - np.mean(values)) ** 2))

    for _ in range(M):
        # Turun başındaki hatalar
        error = y - f
        p = _gbm_sigmoid(f) if loss == "logloss" else None

        # Negatif gradyan
        if loss == "squared":
            residual = error
        elif loss == "absolute":
            residual = np.sign(error)
        elif loss == "huber":
            residual = np.clip(error, -delta, delta)
        else:
            residual = y - p

        def build(indices, depth):
            parent_sse = sse(residual[indices])
            best_sse = parent_sse
            best = None

            if depth < max_depth and len(indices) >= 2:
                for feature in range(X.shape[1]):
                    values = np.unique(X[indices, feature])

                    for a, b in zip(values[:-1], values[1:]):
                        threshold = float(a / 2.0 + b / 2.0)

                        goes_left = (
                            X[indices, feature] <= threshold
                        )
                        left = indices[goes_left]
                        right = indices[~goes_left]

                        if len(left) == 0 or len(right) == 0:
                            continue

                        score = (
                            sse(residual[left])
                            + sse(residual[right])
                        )

                        # Eşitlikte ilk aday korunur.
                        if score < best_sse:
                            best_sse = score
                            best = (
                                feature, threshold, left, right
                            )

            if best is not None and parent_sse - best_sse > 1e-12:
                feature, threshold, left, right = best

                return {
                    "feature": feature,
                    "threshold": threshold,
                    "left": build(left, depth + 1),
                    "right": build(right, depth + 1),
                }

            # Yaprağın son düzeltme değerini hesapla.
            e_leaf = error[indices]

            if loss == "squared":
                value = np.mean(e_leaf)

            elif loss == "absolute":
                value = np.median(e_leaf)

            elif loss == "huber":
                median = np.median(e_leaf)
                value = median + np.mean(
                    np.clip(
                        e_leaf - median,
                        -delta,
                        delta
                    )
                )

            else:
                p_leaf = p[indices]
                value = np.sum(y[indices] - p_leaf) / (
                    np.sum(p_leaf * (1.0 - p_leaf))
                    + 1e-12
                )

            return {"value": float(value)}

        tree = build(np.arange(len(y)), 0)
        model["trees"].append(tree)

        # Ağacın düzeltmesini mevcut tahminlere ekle.
        f += nu * _gbm_tree_predict(tree, X)

        # Güncellemeden sonraki ortalama eğitim kaybı
        error = y - f

        if loss == "squared":
            mean_loss = np.mean(0.5 * error ** 2)

        elif loss == "absolute":
            mean_loss = np.mean(np.abs(error))

        elif loss == "huber":
            mean_loss = np.mean(
                np.where(
                    np.abs(error) <= delta,
                    0.5 * error ** 2,
                    delta * (
                        np.abs(error) - delta / 2.0
                    ),
                )
            )

        else:
            mean_loss = np.mean(
                np.logaddexp(
                    0.0,
                    (1.0 - 2.0 * y) * f
                )
            )

        losses.append(float(mean_loss))

    return model, losses


def gbm_predict(model, X):
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, model["n_features"])

    if X.ndim != 2 or X.shape[1] != model["n_features"]:
        raise ValueError("Ozellik sayisi modelle uyusmali.")

    scores = np.full(
        X.shape[0], model["f0"], dtype=float
    )

    for tree in model["trees"]:
        scores += (
            model["nu"] * _gbm_tree_predict(tree, X)
        )

    return scores