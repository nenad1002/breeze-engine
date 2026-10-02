"""Engine-independent NumPy helpers for numerical validation."""
import numpy as np


def compare_logits(actual, expected, min_cosine):
    """Compare finite, shape-matched logits using the minimum row cosine."""
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Reference/candidate logits have incompatible shapes or non-finite entries")
    a = actual.astype(np.float64).reshape(-1, actual.shape[-1])
    b = expected.astype(np.float64).reshape(a.shape)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    similarity = np.divide((a * b).sum(axis=1), denom, out=np.zeros(len(a)), where=denom != 0)
    similarity[denom == 0] = np.all(a[denom == 0] == b[denom == 0], axis=1)
    cosine = float(similarity.min())
    return {"min_row_cosine": cosine, "max_abs": float(np.max(np.abs(a - b))),
            "last_token_match": bool(actual[-1].argmax() == expected[-1].argmax()),
            "passed": cosine >= min_cosine}