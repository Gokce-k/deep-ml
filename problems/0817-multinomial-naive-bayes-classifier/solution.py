import numpy as np


def multinomial_naive_bayes(
    X_train, y_train, X_test, alpha=1.0
):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)

    # Sıralanmış benzersiz sınıf etiketleri
    classes = np.unique(y_train)

    n_classes = len(classes)
    n_features = X_train.shape[1]

    log_priors = np.zeros(n_classes)
    log_feature_probs = np.zeros(
        (n_classes, n_features)
    )

    for i, label in enumerate(classes):
        # Yalnızca bu sınıfa ait eğitim satırları
        X_class = X_train[y_train == label]

        # Sınıfın öncül olasılığı
        prior = len(X_class) / len(X_train)
        log_priors[i] = np.log(prior)

        # Her özelliğin bu sınıftaki toplam sayısı
        feature_counts = np.sum(X_class, axis=0)

        # Laplace düzeltmeli özellik olasılıkları
        feature_probs = (
            (feature_counts + alpha)
            / (
                np.sum(feature_counts)
                + alpha * n_features
            )
        )

        log_feature_probs[i] = np.log(feature_probs)

    # Her test örneği için her sınıfın log-skoru
    scores = X_test @ log_feature_probs.T + log_priors

    # En yüksek skorlu sütunu gerçek sınıf etiketine çevir.
    return classes[np.argmax(scores, axis=1)]