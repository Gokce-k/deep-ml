import numpy as np

def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    losses = np.maximum(0.0, 1.0 - y_true * y_pred)
    return round(float(np.mean(losses)), 4)