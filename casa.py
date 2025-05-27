import pandas as pd
from pathlib import Path

from modelo import (
    entreno_RandomForestRegressor,
    error_analisis,
    feature_importance,
    model_params_default,
    dalewacho,
    root_mean_squared_error,
)
from utils import nans_x_columna, figura_barrios_geo, comparo_test_train
from geo import (
    me_fijo_si_barrio_esta_bien,
    releno_l3_con_barrio_oficial,
    relleno_latlon_con_media_barrio,
    invierto_lat_lon,
    barrios_con_OSM,
    creo_zonas_mas_precisas
)
from transformaciones import (
    convierto_pesos_a_dolares,
    convierto_ph,
    aplico_transformaciones,
    precio_xmxbxp,
)
from correcciones import (
    correcciones_de_pedo_test,
    correcciones_de_pedo_train,
    mal_values_a_nan_CASA,
    tiro_muchos_nans,
    imputo_ambientes_casa,
    imputo_sfc_casa,
)
from outliers import tiro_outliers_casa
from texto import rooms_from_text

import re

data_dir = Path(
    "/home/mfeijoo/Documents/yo/master/dm/fcen-dm-2025-prediccion-precio-de-propiedades"
)

df_train = pd.read_csv(
    data_dir.joinpath("entrenamiento/entrenamiento.csv"), index_col="id"
)
df_test = pd.read_csv(data_dir.joinpath("a_predecir.csv"), index_col="id")


####### FILTROS #######
def filtro_duplicados(df: pd.DataFrame, subset=["title", "description"], tipo="train"):
    df_out = df.copy()
    df_out = df_out.sort_values(by="start_date")
    # Primero tiro los que se publicaron el mismo día varias veces
    if tipo == "train":
        df_out = df_out.drop_duplicates(
            subset=["start_date", "end_date"] + subset, keep="last"
        )

    # Calculo cuántas veces se repite cada grupo
    total_counts = df_out.groupby(subset)[subset[0]].transform("count")

    # Identifico filas repetidas (más de una vez)
    es_repetido = total_counts > 1

    # Elimino filas que son repetidas y tienen price NaN
    if tipo == "train":
        df_out = df_out[~(es_repetido & df_out["price"].isna())].copy()

    # Ahora cuento la cantidad de duplicados que hay en el subset
    total_counts = df_out.groupby(subset)[subset[0]].transform("count")

    df_out["nro_publicaciones"] = total_counts.values
    if tipo == "train":
        # Keep only the last occurrence of each duplicate group
        mask = ~df_out.duplicated(subset=subset, keep="last")
        df_out = df_out[mask].copy()
    return df_out


def es_efectivamente_casa(df: pd.DataFrame):
    df_out = df.copy()
    regex = r"\bcasa\b|\bph\b|\bchalet\b"
    es_casa = df_out["title"].str.contains(
        regex, flags=re.IGNORECASE, regex=True
    ) | df_out["description"].str.contains(regex, flags=re.IGNORECASE, regex=True)
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
    df_out = df_out.drop(
        columns=["l1", "l2", "operation_type", "ad_type", "l5", "l6", "price_period"]
    )
    return df_out


# %%
# flujo

control = {
    "filtroDatos": True,
    "filtroDuplicados": True,
    "pesosADolares": True,
    "dePedo": True,
    "inviertoLatLon": True,
    "barrioOK": True,
    "barriosMejorados": True,
    "dropGeoNans": True,
    "fillL3cLatlon": True,
    "fillLatloncL3": True,
    "phACasa": True,
    "filtroCasa": True,
    "trans": True,
    "mal2nan": True,
    "tiroNans": False,
    "imputAmbs": True,
    "imputSfc": True,
    "tiroOutliers": True,
    "precio_xbxmxp": True,
}

# %%
######## OPERACIONES COMUNES A TODOS ########
if control.get("filtroDatos", True):
    # Filtro por Argentina, CABA y Venta
    train = filtro_datos(df_train, tipo="train")
    test = filtro_datos(df_test, tipo="test")

if control.get("filtroDuplicados", True):
    # Filtro duplicados
    train = filtro_duplicados(train, tipo="train")
    test = filtro_duplicados(test, tipo="test")

if control.get("pesosADolares", True):
    # Convierto pesos a dolares (solo train)
    train = convierto_pesos_a_dolares(train, debug=False)

if control.get("dePedo", True):
    # Tiro cosas fortuitas que fui viendo
    train = correcciones_de_pedo_train(train)
    test = correcciones_de_pedo_test(test)

######### LAT LON L3 #########
if control.get("inviertoLatLon", True):
    # Primero invierto Lat lon porque están al revés
    train = invierto_lat_lon(train)
    test = invierto_lat_lon(test)

if control.get("barrioOK", True):
    # Me fijo si el barrio coincide con BA DATA
    train = me_fijo_si_barrio_esta_bien(train)
    test = me_fijo_si_barrio_esta_bien(test)

if control.get("dropGeoNans", True):
    # Donde tengo lat lon l3 Nan, lo tiro
    train = train.dropna(subset=["lat", "lon", "l3"])

if control.get("fillL3cLatlon", True):
    # Donde tengo latlon pero no l3, completo con barrio_oficial
    train = releno_l3_con_barrio_oficial(train)
    test = releno_l3_con_barrio_oficial(test)

if control.get("fillLatloncL3", True):
    # Donde tengo l3 pero no latlon, completo con la media de lat lon de ese l3
    train = relleno_latlon_con_media_barrio(train, by="l3")
    test = relleno_latlon_con_media_barrio(test, by="l3")


if control.get("barriosMejorados", True):
    train = creo_zonas_mas_precisas(train, uso_osm=True)
    test = creo_zonas_mas_precisas(test, uso_osm=True)

# figura_barrios_geo(train, tipo="train", by="l3", show=False)

# %%
###### SEPARO POR TIPO DE PROPIEDAD #######
if control.get("phACasa", True):
    # convierto Ph a Casa
    train = convierto_ph(train, debug=False)
    test = convierto_ph(test, debug=False)

if control.get("filtroCasa", True):
    # Filtro por Casa
    casa_train = filtro_por_propiedad(
        train, property_type="Casa", busco_en_titulo=False
    )
    casa_test = filtro_por_propiedad(test, property_type="Casa", busco_en_titulo=False)

if control.get("trans", True):
    # Aplico transformaciones
    casa_train = aplico_transformaciones(casa_train)
    casa_test = aplico_transformaciones(casa_test)

# %%
####### AMBIENTES #######
if control.get("mal2nan", True):
    # Pongo Nans donde no tiene sentido el valor
    casa_train = mal_values_a_nan_CASA(casa_train, debug=False)
    casa_test = mal_values_a_nan_CASA(casa_test, debug=False)

if control.get("tiroNans", True):
    # Tiro las filas que tienen muchos nans
    casa_train = tiro_muchos_nans(
        casa_train,
        debug=False,
        tol=2,
        columnas_relevantes=[
            "rooms",
            "bedrooms",
            "bathrooms",
            "surface_total",
            "surface_covered",
        ],
    )

if control.get("imputAmbs", True):
    # Imputo los nans
    casa_train = imputo_ambientes_casa(
        casa_train, tipo="train", debug=False, imputo=["rooms"]
    )  # con rooms solo funciona mejor
    casa_test = imputo_ambientes_casa(
        casa_test, tipo="test", debug=False, imputo=["rooms"]
    )


if control.get("imputSfc", True):
    # Imputo los nans
    casa_train = imputo_sfc_casa(
        casa_train, tipo="train", debug=False, imputo=["surface_covered"]
    )
    casa_test = imputo_sfc_casa(
        casa_test, tipo="test", debug=False, imputo=["surface_covered"]
    )

if control.get("tiroOutliers", True):
    # Tiro outliers
    casa_train = tiro_outliers_casa(casa_train, casa_test, debug=False)


####### PRECIO #######
if control.get("precio_xbxmxp", True):
    # Calculo precio por m2
    casa_train = precio_xmxbxp(
        casa_train, casa_test, by="l3", sub=False, tipo="train", debug=False
    )
    casa_test = precio_xmxbxp(casa_train, casa_test, by="l3", sub=False, tipo="test", debug=False)

print("asd")
# %%
nombre_prueba = "_".join([k for k, v in control.items() if v == True])

####### ACOMODO #######
predictores_potenciales = [
    "surface_covered",
    "surface_total",
    "rooms",
    "bedrooms",
    "bathrooms",
    "lat",
    "lon",
    "nro_publicaciones",
    "en_pesos",
    "is_barrio_ok",
    "es_PH",
    "sfc_x_bath",
    "sup_x_room",
    "total-cov",
    "room-bed",
    "corrijo_rooms",
    "corrijo_bedrooms",
    "corrijo_bathrooms",
    "categoria",
    "imputo_surface_total",
    "imputo_surface_covered",
    "inverti_sups",
]

predictores = list(
    set(predictores_potenciales)
    & set(list(casa_train.columns))
    & set(list(casa_test.columns))
)

target = "price"

train = casa_train[predictores + [target]]


X = train.dropna(subset=predictores + [target])[predictores]
y = train.dropna(subset=predictores + [target])[target]

####### ENTRENO #######
reg, y_pred, score_test, (X_test, y_test) = entreno_RandomForestRegressor(
    X=X,
    y=y,
    test_size=0.1,
    random_state=42,
    model_params=model_params_default,
)

importancia = feature_importance(reg, predictores)

errores = error_analisis(X_test, y_test, y_pred, by="error")


# %%
### VOY CON TEST

reg, y_pred, X_test = dalewacho(
    train=casa_train,
    test=casa_test,
    predictores=predictores,
    target=target,
    random_state=42,
    model_params=model_params_default,
)

# Comparo con Facu
fac = pd.read_csv(
    "/home/mfeijoo/Documents/yo/master/dm/fcen-dm-2025-prediccion-precio-de-propiedades/predicciones_facu.csv",
    index_col="id",
)

fac = fac.loc[X_test.index]

fac["prediccion"] = y_pred

fac["error"] = fac["price"] - fac["prediccion"]
fac["error_abs"] = abs(fac["error"])
fac["error_rel"] = fac["error"] / fac["price"]
rmse = root_mean_squared_error(fac["price"], fac["prediccion"])
print(nombre_prueba)
print(f"RMSE: {rmse}")
# %%


print("asd")
