from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ])

def hyperparameter_search(X_train, y_train):
    pipeline = build_pipeline()

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid
