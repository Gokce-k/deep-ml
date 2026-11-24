import numpy as np

def linear_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    iterations: int
) -> np.ndarray:

    m, n = X.shape
    theta = np.zeros(n)  # Bütün ağırlıkları sıfırdan başlat

    for _ in range(iterations):
        predictions = X @ theta          # Tahminler
        errors = predictions - y         # Tahmin hataları
        gradient = (X.T @ errors) / m     # Kaybın gradyanı
        theta = theta - alpha * gradient  # Ağırlıkları güncelle

    return theta