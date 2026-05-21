import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# 引入你已有的全部模型
from regressors import (
    MLPRegressorTorch,
    SLPRegressorTorch,
    RFRegressor,
    XGBRegressorWrapper,
    RidgeRegressor,
    LassoRegressor,
    PLSRRegressor
)

# 1. 造一份简单回归数据
X, y = make_regression(n_samples=300, n_features=20, noise=0.1, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 要测试的所有模型（按需增删）
models = {
    "MLP": MLPRegressorTorch(layers=2, neurons=10, epochs=5, device="cpu"),
    "SLP": SLPRegressorTorch(neurons=10, epochs=5, device="cpu"),
    "RF":  RFRegressor(n_estimators=50, random_state=42),
    "XGB": XGBRegressorWrapper(n_estimators=50, random_state=42, verbosity=0),
    "Ridge": RidgeRegressor(alpha=1.0),
    "Lasso": LassoRegressor(alpha=0.01),
    "PLSR":  PLSRRegressor(n_components=3)
}

# 3. 遍历测试
for name, model in models.items():
    print(f"Testing {name} ... ", end="")
    try:
        model.fit(X_tr, y_tr)               # 训练
        pred = model.predict(X_te)         # 预测
        assert pred.shape == y_te.shape    # 维度检查
        rmse = np.sqrt(np.mean((pred - y_te) ** 2))
        print(f"OK  (RMSE={rmse:.3f})")
    except Exception as e:
        print("FAIL")
        print(e)