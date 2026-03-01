from sklearn.metrics import roc_auc_score

def evaluate(model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, prob)
