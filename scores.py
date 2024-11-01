from numpy import corrcoef, double, log, ix_
from numpy.linalg import cholesky


class BIC:

    def __init__(self, X, pd=2.0):
        self.n, self.p = X.shape
        self.pd = pd
        self.corr = corrcoef(X.T, dtype=double)
        self.w = pd / 4.0 * log(self.n) / self.n

    def get_n(self): return self.n

    def get_p(self): return self.p

    def get_pd(self): return self.pd

    def set_pd(self, pd): 
        self.pd = pd
        self.w = pd / 4.0 * log(self.n) / self.n

    def score(self, x, Z=None):
        if Z is None: Z = []
        S = sorted(Z) + [x]
        L = cholesky(self.corr[ix_(S, S)])
        return - log(L[-1, -1]) - len(Z) * self.w