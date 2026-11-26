import numpy as np


def train_logreg(X, y, learning_rate, iterations):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # İlk sütun bias için 1'lerden oluşur.
    X = np.column_stack((np.ones(X.shape[0]), X))

    # theta = [bias, w1, w2, ...]
    theta = np.zeros(X.shape[1], dtype=float)
    losses = []

    for _ in range(iterations):
        z = X @ theta

        # Sayısal olarak kararlı sigmoid hesabı
        preds = np.exp(-np.logaddexp(0.0, -z))

        err = preds - y

        # Toplam binary cross-entropy
        loss = np.sum(np.logaddexp(0.0, z) - y * z)

        # Güncellemeden önceki kaybı kaydet.
        losses.append(float(loss))

        gradient = X.T @ err
        theta -= learning_rate * gradient

    coefficients = np.round(theta, 4).tolist()
    losses = np.round(losses, 4).tolist()

    return coefficients, losses