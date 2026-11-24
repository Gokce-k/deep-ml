import numpy as np

def svm_margin_width(w: np.ndarray) -> float:
    return float(2.0 / np.linalg.norm(w))