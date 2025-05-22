import pandas as pd
from pathlib import Path

from modelo import entreno_RandomForestRegressor, error_analisis, feature_importance, model_params_default
from utils import nans_x_columna
from transformaciones import convierto_pesos_a_dolares, convierto_ph

import re

data_dir = Path("/home/mfeijoo/Documents/yo/master/dm/fcen-dm-2025-prediccion-precio-de-propiedades")

df_train = pd.read_csv(data_dir.joinpath("entrenamiento/entrenamiento.csv"), index_col="id")
df_test = pd.read_csv(data_dir.joinpath("a_predecir.csv"), index_col="id")




####### FILTROS #######

def es_efectivamente_casa(df):
    df_out = df.copy()
    regex = r"\bcasa\b|\bph\b|\bchalet\b" 
    df_out["title"].str.contains(regex, flags=re.IGNORECASE, regex=True)
    
    return df_out


def filtro_datos(df: pd.DataFrame, tipo="train", property_type="Casa"):
    df_out = df.copy()
    df_out = df_out[df_out["property_type"].isin([property_type])]
    df_out = convierto_ph(df_out, convierto_a="Casa")
    if tipo == "train":
        df_out = df_out[df_out["l1"] == "Argentina"]
        df_out = df_out[df_out["l2"] == "Capital Federal"]
        df_out = df_out[df_out["operation_type"] == "Venta"]
        df_out = convierto_pesos_a_dolares(df_out)
        
    return df_out

casa_train = filtro_datos(df_train, tipo="train", property_type="Casa")
casa_test = filtro_datos(df_test, tipo="test", property_type="Casa")

####### ACOMODO #######

predictores = [
    "surface_covered",
    "rooms",
    "bedrooms", 
    "bathrooms",
    "lat",
    "lon",
]

target = "price"

train = df_train[predictores + [target]]
X = train.dropna(subset=predictores + [target])[predictores]
y = train.dropna(subset=predictores + [target])[target]

####### ENTRENO #######
reg, score_test, y_pred, (X_test, y_test) = entreno_RandomForestRegressor(
    X=X, y=y, test_size=0.1, random_state=42, model_params=model_params_default
)

feature_importance(reg, predictores)

errores = error_analisis(X_test, y_test, y_pred, by="error")