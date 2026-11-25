import numpy as np


def soft_voting_classifier(probabilities, weights=None):
    probabilities = np.asarray(probabilities, dtype=float)

    n_classifiers, n_samples, n_classes = probabilities.shape

    if weights is None:
        weights = np.ones(n_classifiers, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # Ağırlıkların toplamını 1 yap.
    weights = weights / np.sum(weights)

    average_probabilities = np.zeros(
        (n_samples, n_classes),
        dtype=float
    )

    # Her modelin ağırlıklı olasılıklarını ekle.
    for j in range(n_classifiers):
        average_probabilities += (
            weights[j] * probabilities[j]
        )

    # Her örnek için en yüksek olasılıklı sınıfı seç.
    return np.argmax(average_probabilities, axis=1).tolist()