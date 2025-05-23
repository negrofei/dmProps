import pandas as pd
from pathlib import Path

from modelo import entreno_RandomForestRegressor, error_analisis, feature_importance, model_params_default
from utils import nans_x_columna, figura_barrios_geo
from geo import me_fijo_si_barrio_esta_bien, releno_l3_con_barrio_oficial, relleno_latlon_con_media_barrio, invierto_lat_lon
from transformaciones import convierto_pesos_a_dolares, convierto_ph

import re

data_dir = Path("/home/mfeijoo/Documents/yo/master/dm/fcen-dm-2025-prediccion-precio-de-propiedades")

df_train = pd.read_csv(data_dir.joinpath("entrenamiento/entrenamiento.csv"), index_col="id")
df_test = pd.read_csv(data_dir.joinpath("a_predecir.csv"), index_col="id")




####### FILTROS #######
def filtro_duplicados(df: pd.DataFrame, subset=["title", "description"]):
    df_out = df.copy()
    # Primero tiro los que se publicaron el mismo día varias veces
    df_out = df_out.drop_duplicates(subset=["start_date", "end_date"] + subset, keep="last")

    # Calculo cuántas veces se repite cada grupo
    total_counts = df_out.groupby(subset)[subset[0]].transform('count')

    # Identifico filas repetidas (más de una vez)
    es_repetido = total_counts > 1
    
    # Elimino filas que son repetidas y tienen price NaN
    df_out = df_out[~(es_repetido & df_out["price"].isna())].copy()
    
    # Ahora cuento la cantidad de duplicados que hay en el subset
    counts = df_out.groupby(subset).cumcount(ascending=True)
    total_counts = df_out.groupby(subset)[subset[0]].transform('count')
    
    df_out["nro_publicaciones"] = total_counts.values
    # Keep only the last occurrence of each duplicate group
    mask = ~df_out.duplicated(subset=subset, keep='last')
    df_out = df_out[mask].copy()
    
    return df_out

def es_efectivamente_casa(df: pd.DataFrame):
    df_out = df.copy()
    regex = r"\bcasa\b|\bph\b|\bchalet\b" 
    es_casa = (
        df_out["title"].str.contains(regex, flags=re.IGNORECASE, regex=True) | \
        df_out["description"].str.contains(regex, flags=re.IGNORECASE, regex=True)
    )
    return es_casa

def filtro_por_propiedad(df: pd.DataFrame, property_type="Casa", busco_en_titulo=False):
    df_out = df.copy()
    df_out = df_out[df_out["property_type"] == property_type]
    if busco_en_titulo:
        resto = df[~df.index.isin(df_out.index)].copy()
        casa_en_resto = es_efectivamente_casa(resto)
        aver = resto[casa_en_resto]
        raise NotImplementedError("No implementado")
    df_out = df_out.drop(columns=["property_type"])
    return df_out

def filtro_datos(df: pd.DataFrame, tipo="train"):
    df_out = df.copy()
    if tipo == "train":
        df_out = df_out[df_out["l1"] == "Argentina"]
        df_out = df_out[df_out["l2"] == "Capital Federal"]
        df_out = df_out[df_out["operation_type"] == "Venta"]
    df_out = df_out.drop(columns=["l1", "l2", "operation_type", "ad_type", "l5", "l6", "price_period"])
    return df_out

#%%
######## OPERACIONES COMUNES A TODOS ########
# Filtro por Argentina, CABA y Venta
train = filtro_datos(df_train, tipo="train")
test = filtro_datos(df_test, tipo="test")

train = filtro_duplicados(train)

# Convierto pesos a dolares
train = convierto_pesos_a_dolares(train)

######### LAT LON L3 #########
# Primero invierto Lat lon porque están al revés
train = invierto_lat_lon(train)
test = invierto_lat_lon(test)

# Me fijo si el barrio coincide con BA DATA
train = me_fijo_si_barrio_esta_bien(train)
test = me_fijo_si_barrio_esta_bien(test)

# Donde tengo lat lon l3 Nan, lo tiro
train = train.dropna(subset=["lat", "lon", "l3"])

# Donde tengo latlon pero no l3, completo con barrio_oficial
train = releno_l3_con_barrio_oficial(train)
test = releno_l3_con_barrio_oficial(test)

# Donde tengo l3 pero no latlon, completo con la media de lat lon de ese l3
train = relleno_latlon_con_media_barrio(train, by="barrio_oficial")
test = relleno_latlon_con_media_barrio(test, by="barrio_oficial")

figura_barrios_geo(train, tipo="train", by="l3", show=False)

#%%
###### SEPARO POR TIPO DE PROPIEDAD #######
# convierto Ph a Casa
train = convierto_ph(train)
test = convierto_ph(test)

# Filtro por Casa
casa_train = filtro_por_propiedad(train, property_type="Casa", busco_en_titulo=False)
casa_test = filtro_por_propiedad(test, property_type="Casa", busco_en_titulo=False)


print('asd')

####### ACOMODO #######

predictores = [
    "surface_covered",
    "rooms",
    "bedrooms", 
    "bathrooms",
    "lat",
    "lon",
    "nro_publicaciones",
    "en_pesos",
    "is_barrio_ok",
]

target = "price"

train = casa_train[predictores + [target]]
X = train.dropna(subset=predictores + [target])[predictores]
y = train.dropna(subset=predictores + [target])[target]

####### ENTRENO #######
reg, score_test, y_pred, (X_test, y_test) = entreno_RandomForestRegressor(
    X=X, y=y, test_size=0.1, random_state=42, model_params=model_params_default
)

feature_importance(reg, predictores)

errores = error_analisis(X_test, y_test, y_pred, by="error")

#%%
print('asd')