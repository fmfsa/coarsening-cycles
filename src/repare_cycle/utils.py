import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy.stats import f


class SimpleCanCorr:
    def __init__(self, X, Y, Z=None, tol=1e-8):
        if X.ndim == 1:
            X = X[:, None]
        if Y.ndim == 1:
            Y = Y[:, None]
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows (observations)")

        # Center X and Y
        X = X - X.mean(0)
        Y = Y - Y.mean(0)

        # Handle Z (Conditioning set)
        kz = 0
        if Z is not None:
            if Z.ndim == 1:
                Z = Z[:, None]
            if Z.shape[0] != X.shape[0]:
                raise ValueError("Z must have the same number of rows as X and Y")

            # Center Z
            Z = Z - Z.mean(0)
            kz = Z.shape[1]

            # Residualize X and Y on Z
            Q, _ = np.linalg.qr(Z)
            X = X - Q @ (Q.T @ X)
            Y = Y - Q @ (Q.T @ Y)

        self.nobs = X.shape[0] - kz
        self.kx = X.shape[1]
        _, self.ky = Y.shape

        # Compute SVDs
        Ux, sx, Vx = svd(X, full_matrices=False)
        Uy, sy, Vy = svd(Y, full_matrices=False)
        if np.any(sx <= tol):
            raise ValueError("X is collinear")
        if np.any(sy <= tol):
            raise ValueError("Y is collinear")

        # Build decorrelated projections
        Vx_ds = Vx.T / sx
        Vy_ds = Vy.T / sy

        # Singular values of cross-covariance
        U, s, V = svd(Ux.T @ Uy, full_matrices=False)

        self.cancorr = np.clip(s, 0, 1)
        self.x_cancoef = Vx_ds @ U
        self.y_cancoef = Vy_ds @ V.T

    def wilks_lambda_test(self):
        """Return Wilks' Lambda test stats for canonical correlations."""
        eigenvals = self.cancorr**2
        k_yvar = self.ky
        k_xvar = self.kx
        nobs = self.nobs

        stats = []
        prod = 1.0
        for i in range(len(eigenvals) - 1, -1, -1):
            prod *= 1 - eigenvals[i]
            p = k_yvar - i
            q = k_xvar - i
            r = (nobs - k_yvar - 1) - (p - q + 1) / 2
            u = (p * q - 2) / 4
            df1 = p * q
            if p**2 + q**2 - 5 > 0:
                t = np.sqrt(((p * q) ** 2 - 4) / (p**2 + q**2 - 5))
            else:
                t = 1
            df2 = r * t - 2 * u
            lmd = prod ** (1 / t)
            F_stat = (1 - lmd) / lmd * df2 / df1
            pval = f.sf(F_stat, df1, df2)
            stats.append((self.cancorr[i], prod, df1, df2, F_stat, pval))

        return pd.DataFrame(
            stats[::-1],
            columns=[
                "Canonical Correlation",
                "Wilks' lambda",
                "Num DF",
                "Den DF",
                "F Value",
                "Pr > F",
            ],
        )
