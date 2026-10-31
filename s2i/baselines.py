"""
Lagged ridge-regression baseline (within-subject, trained on the SAME calibration
trials the deep model gets). The old paper compared against numbers reported in
other papers on other data; a claim of improvement is only meaningful against a
baseline run on the identical split.
"""
import numpy as np

DEFAULT_LAGS = (-40, -20, -10, -5, 0, 5, 10, 20, 40)   # samples @200 Hz = +-200 ms


def lagged(e, lags=DEFAULT_LAGS):
    C, L = e.shape
    out = np.zeros((C * len(lags), L), dtype=np.float32)
    for j, lag in enumerate(lags):
        blk = out[j * C:(j + 1) * C]
        if lag > 0:
            blk[:, lag:] = e[:, :-lag]
        elif lag < 0:
            blk[:, :lag] = e[:, -lag:]
        else:
            blk[:] = e
    return out


def _stack(trials_e, trials_y, lags):
    X = np.concatenate([lagged(e, lags) for e in trials_e], 1).T
    Y = np.concatenate(trials_y, 1).T
    return X.astype(np.float64), Y.astype(np.float64)


def _r_cols(P, T):
    P = P - P.mean(0)
    T = T - T.mean(0)
    return (P * T).sum(0) / (np.sqrt((P * P).sum(0) * (T * T).sum(0)) + 1e-12)


class LaggedRidge:
    def __init__(self, lags=DEFAULT_LAGS, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)):
        self.lags, self.alphas = lags, alphas

    def _solve(self, X, Y):
        mu, sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - mu) / sd
        ym = Y.mean(0)
        G = Xs.T @ Xs
        lam, Q = np.linalg.eigh(G)
        QtXtY = Q.T @ (Xs.T @ (Y - ym))
        return mu, sd, ym, lam, Q, QtXtY

    def fit(self, trials_e, trials_y, val_frac=0.2):
        n = len(trials_e)
        n_tr = max(1, int(round(n * (1 - val_frac))))
        scale = None
        if n - n_tr >= 1:
            X, Y = _stack(trials_e[:n_tr], trials_y[:n_tr], self.lags)
            Xv, Yv = _stack(trials_e[n_tr:], trials_y[n_tr:], self.lags)
            mu, sd, ym, lam, Q, QtXtY = self._solve(X, Y)
            scale = lam.mean()
            Xvs = (Xv - mu) / sd
            scores = []
            for a in self.alphas:
                W = Q @ (QtXtY / (lam + a * scale)[:, None])
                scores.append(_r_cols(Xvs @ W + ym, Yv))
            best = np.argmax(np.stack(scores), 0)          # per-contact alpha
        X, Y = _stack(trials_e, trials_y, self.lags)
        mu, sd, ym, lam, Q, QtXtY = self._solve(X, Y)
        scale = lam.mean()
        if n - n_tr < 1:
            best = np.full(Y.shape[1], self.alphas.index(10) if 10 in self.alphas else 0)
        W = np.zeros((X.shape[1], Y.shape[1]))
        for ai, a in enumerate(self.alphas):
            cols = np.where(best == ai)[0]
            if len(cols):
                W[:, cols] = Q @ (QtXtY[:, cols] / (lam + a * scale)[:, None])
        self.mu, self.sd, self.ym, self.W = mu, sd, ym, W
        return self

    def predict(self, trials_e):
        out = []
        for e in trials_e:
            X = (lagged(e, self.lags).T - self.mu) / self.sd
            out.append((X @ self.W + self.ym).T.astype(np.float32))
        return out
