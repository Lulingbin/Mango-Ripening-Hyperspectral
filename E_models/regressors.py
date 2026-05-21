import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression

# ---------- 工具 ----------
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# ---------- 基类 ----------
class BaseRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError
    def cv_score(self, X, y, cv=3, scoring=rmse_scorer):
        return GridSearchCV(
            self, {}, cv=cv, scoring=scoring, n_jobs=-1
        ).fit(X, y).cv_results_["mean_test_score"].mean()

# ---------- 1. MLP ----------
class MLPRegressorTorch(BaseRegressor):
    def __init__(self,
                 layers=2,
                 neurons=50,
                 lr=0.05,
                 epochs=100,
                 batch_size=32,
                 device="cpu"):
        super().__init__()
        self.layers = layers
        self.neurons = neurons
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def _build_model(self, n_features):
        seq = [nn.Linear(n_features, self.neurons), nn.ReLU()]
        for _ in range(self.layers - 1):
            seq += [nn.Linear(self.neurons, self.neurons), nn.ReLU()]
        seq += [nn.Linear(self.neurons, 1)]
        self.net_ = nn.Sequential(*seq).to(self.device)
        self.optimizer_ = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        self.criterion_ = nn.MSELoss()

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)

        self._build_model(X.shape[1])
        self.net_.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(X.size(0))
            for i in range(0, X.size(0), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                self.optimizer_.zero_grad()
                outputs = self.net_(X[indices])
                loss = self.criterion_(outputs, y[indices])
                loss.backward()
                self.optimizer_.step()
        return self

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.net_(X).cpu().numpy().ravel()

# ---------- 2. SLP ----------
class SLPRegressorTorch(BaseRegressor):
    def __init__(self,
                 neurons=50,
                 lr=0.01,
                 epochs=100,
                 batch_size=32,
                 device="cpu"):
        super().__init__()
        self.neurons = neurons
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)

        self.net_ = nn.Sequential(
            nn.Linear(X.shape[1], self.neurons),
            nn.ReLU(),
            nn.Linear(self.neurons, 1)
        ).to(self.device)
        self.optimizer_ = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        self.criterion_ = nn.MSELoss()

        self.net_.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(X.size(0))
            for i in range(0, X.size(0), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                self.optimizer_.zero_grad()
                outputs = self.net_(X[indices])
                loss = self.criterion_(outputs, y[indices])
                loss.backward()
                self.optimizer_.step()
        return self

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.net_(X).cpu().numpy().ravel()

# ---------- 3. RandomForestRegressor ----------
class RFRegressor(BaseRegressor):
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ---------- 4. XGBoost ----------
class XGBRegressorWrapper(BaseRegressor):
    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)




# ---------- 5. Ridge Regression ----------
class RidgeRegressor(BaseRegressor):
    def __init__(self, alpha=1.0, **kwargs):
        """
        alpha: L2 正则强度，等价于 1/(2*C)。
               若想与原文网格对应，可设 alpha ∈ [1e-3, 1e-2, 0.1, 1, 10, 100]
        """
        self.model = Ridge(alpha=alpha, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# ---------- 6. Lasso Regression ----------
class LassoRegressor(BaseRegressor):
    def __init__(self, alpha=1.0, **kwargs):
        """
        alpha: L1 正则强度。
               可设 alpha ∈ [1e-4, 1e-3, 1e-2, 0.1, 1]
        """
        self.model = Lasso(alpha=alpha, max_iter=10000, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)




# ---------- 7. PLSR ----------
class PLSRRegressor(BaseRegressor):
    def __init__(self, n_components=2, scale=True, max_iter=500, tol=1e-06):
        """
        n_components: 偏最小二乘成分个数（相当于潜变量个数）
        scale: 是否对 X 和 y 做标准化
        max_iter: 迭代最大次数
        tol: 收敛容差
        """
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.model = PLSRegression(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X).ravel()