"""AHP (absolute-measurement) and TOPSIS comparators on the same decision matrix as the FDSS.
The AHP pairwise matrix is the ILLUSTRATIVE matrix of the original repository (not elicited)."""
import numpy as np

CRITERIA = ["safety", "ease", "height", "reuse", "cost"]
BENEFIT = np.array([1, 1, 1, 1, 0], dtype=bool)
PAIRWISE = np.array([[1, 3, 2, 4, 5], [1/3, 1, 1/2, 2, 3], [1/2, 2, 1, 3, 4],
                     [1/4, 1/2, 1/3, 1, 2], [1/5, 1/3, 1/4, 1/2, 1]])
RI = {3: .58, 4: .90, 5: 1.12}

def ahp_weights(M=PAIRWISE):
    vals, vecs = np.linalg.eig(M); i = np.argmax(vals.real)
    w = vecs[:, i].real; w = w / w.sum(); n = M.shape[0]
    cr = ((vals.real.max() - n) / (n - 1)) / RI[n]
    return w, float(cr)

def ahp_scores(X, w):
    X = np.asarray(X, float)
    N = np.where(BENEFIT, X / X.max(0), X.min(0) / X)
    return N @ w

def topsis_scores(X, w):
    X = np.asarray(X, float)
    R = X / np.sqrt((X ** 2).sum(0)); V = R * w
    ideal = np.where(BENEFIT, V.max(0), V.min(0)); anti = np.where(BENEFIT, V.min(0), V.max(0))
    dp = np.sqrt(((V - ideal) ** 2).sum(1)); dn = np.sqrt(((V - anti) ** 2).sum(1))
    return dn / (dp + dn + 1e-12)
