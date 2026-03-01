import os
import pickle
from src.data import load_data, map_target
from src.features import remove_leakage, encode_categoricals
from src.model import hyperparameter_search
from sklearn.model_selection import train_test_split
from src.evaluation import evaluate

DATA_PATH = "data/bank-full-case.csv"
MODEL_PATH = "models/best_model.pkl"

def run_pipeline():
    print("Carregando dados...")
    df = load_data(DATA_PATH)
    df = map_target(df)
    df = remove_leakage(df)
    df = encode_categoricals(df)

    X = df.drop(columns=["y"])
    y = df["y"]

    print("Separando treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Executando busca de hiperparâmetros...")
    grid = hyperparameter_search(X_train, y_train)

    print("Melhores parâmetros:", grid.best_params_)
    print("Melhor AUC:", grid.best_score_)

    print("Avaliando no conjunto de teste...")
    auc_test = evaluate(grid.best_estimator_, X_test, y_test)
    print("AUC Teste:", auc_test)

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(grid.best_estimator_, f)

    print("Modelo salvo com sucesso.")
