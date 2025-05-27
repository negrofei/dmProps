from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from utils import elapsed_time

import pandas as pd
import matplotlib.pyplot as plt

model_params_default = {
    "n_estimators": 500,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": 42,
}

from pathlib import Path
output = Path("/home/mfeijoo/Documents/yo/master/dm/output/")

def entreno_RandomForestRegressor(
    X: pd.DataFrame, y: pd.DataFrame, test_size=0.1, random_state=42, model_params=model_params_default
):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # print("X_train", X_train.shape)
    # print("X_test", X_test.shape)
    # print("y_train", y_train.shape)
    # print("y_test", y_test.shape)

    reg = RandomForestRegressor(**model_params)

    @elapsed_time
    def fit_reg():
        reg.fit(X_train, y_train)

    fit_reg()

    y_pred = reg.predict(X_test)

    score_test = root_mean_squared_error(y_test, y_pred)
    print(f"Score en prueba: {score_test}")
    return reg, y_pred, score_test, (X_test, y_test)


def dalewacho(
    train: pd.DataFrame, test: pd.DataFrame, predictores: list, target: str, random_state=42, model_params=model_params_default
):
    X_train = train.dropna(subset=predictores + [target])[predictores]
    y_train = train.dropna(subset=predictores + [target])[target]

    X_test = test[predictores]
    reg = RandomForestRegressor(**model_params)

    @elapsed_time
    def fit_reg():
        reg.fit(X_train, y_train)

    fit_reg()

    y_pred = reg.predict(X_test)

    return reg, y_pred, X_test


def feature_importance(reg, predictores, show=False, save=True):
    importances = reg.feature_importances_

    # Convertir a DataFrame para ordenar y graficar
    importancia_df = pd.DataFrame(
        {"feature": predictores, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    # print(importancia_df.round(3))

    # Gráfico opcional
    importancia_df.head(20).plot(
        kind="barh",
        x="feature",
        y="importance",
        figsize=(8, 6),
        title="Importancia de Variables",
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(output.joinpath("feature_importance.png"))
    return importancia_df


def error_analisis(X_test, y_test, y_pred, by="error"):
    X_test["error"] = abs(y_pred - y_test)
    X_test["price"] = y_test
    X_test["pred_price"] = y_pred
    X_test["error_relativo"] = X_test["error"] / y_test
    X_test = X_test.sort_values(by=by, ascending=False)
    return X_test
