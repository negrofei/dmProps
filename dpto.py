import pandas as pd
from pathlib import Path
from datetime import datetime

from filtros import filtro_duplicados, filtro_datos, filtro_por_propiedad

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
    creo_zonas_mas_precisas,
)
from transformaciones import (
    convierto_pesos_a_dolares,
    convierto_ph,
    aplico_transformaciones,
    precio_xmxbxp,
    precio_estimado
)
from correcciones import (
    correcciones_de_pedo_test,
    correcciones_de_pedo_train,
    mal2nan_ambientes_dpto,
    tiro_muchos_nans,
    imputo_ambientes_dpto,
    imputo_sfc_dpto,
    imputo_precio,
    mal2nan_sfc_dpto,
    mal2nan_precio_dpto,
    corrijo_superficies_dpto,
)
from outliers import tiro_outliers_dpto
from texto import aplico_dummies_dpto

import numpy as np

data_dir = Path(__file__).parent.joinpath(
    "fcen-dm-2025-prediccion-precio-de-propiedades"
)

df_train = pd.read_csv(
    data_dir.joinpath("entrenamiento/entrenamiento.csv"), index_col="id"
)
df_test = pd.read_csv(data_dir.joinpath("a_predecir.csv"), index_col="id")

# %%
# flujo

control = {
    ######## OPERACIONES COMUNES A TODOS ########
    "filDatos": True,
    "filDup": True,
    "ars2usd": True,
    "dePedo": True,
    ######### LAT LON L3 #########
    "invLatLon": True,
    "barrioOK": True,
    "barrioMej": False,
    "dropGeoNans": True,
    "impL3wBarrioOK": True,
    "impLatloncL3": False,
    "osm": False,
    ###### PROPIEDAD ######
    "ph2dpto": False,
    "filDpto": True,
    "tr": True,
    ####### AMBIENTES #######
    "mal2nan_amb": True,
    "impAmbs": True,
    "ams2imp": ["rooms", "bedrooms", "bathrooms"],
    ####### SUPERFICIE #######
    "mal2nan_sfc": True,
    "impSfc": True,
    "sfc2imp": ["surface_total", "surface_covered", "invierto_surface"],
    "sfcMalDec": True,
    ####### PRECIO #######
    "precio_xbxmxp": True,
    "impPrecio": True,
    "precio_estimado": True,
    ####### OUTLIERS #######
    "dropOut": True,
    "minmax": {
        "rooms": [1, 10],
        "bedrooms": [0, 10],
        "bathrooms": [1, 7],
        "surface_total": [10, 700],
        "surface_covered": [10, 700],
        "price": [10000, np.inf],
        "sup_x_room": [15, 300],
        "room-bed": [0, 10],
        "sfc_x_bath": [10, 300],
        "precio_m2": [500, 10000],
        "precio_xmxbxp": [500, np.inf],
    },
    "otros": {
        "sigma": 1,
        "diff_pm2": 4000,
    },
    ###### TXT ######
    "dums": True,
    
}

# %%
######## OPERACIONES COMUNES A TODOS ########
####### FILTROS #######
if control.get("filDatos", True):
    # Filtro por Argentina, CABA y Venta
    train = filtro_datos(df_train, tipo="train")
    test = filtro_datos(df_test, tipo="test")

if control.get("filDup", True):
    # Filtro duplicados
    train = filtro_duplicados(train, tipo="train")
    test = filtro_duplicados(test, tipo="test")

if control.get("ars2usd", True):
    # Convierto pesos a dolares (solo train)
    train = convierto_pesos_a_dolares(train, debug=False)

if control.get("dePedo", True):
    # Tiro cosas fortuitas que fui viendo
    train = correcciones_de_pedo_train(train)
    test = correcciones_de_pedo_test(test)

######### LAT LON L3 #########
if control.get("invLatLon", True):
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

if control.get("impL3wBarrioOK", True) and control.get("barrioOK", True):
    # Donde tengo latlon pero no l3, completo con barrio_oficial
    train = releno_l3_con_barrio_oficial(train)
    test = releno_l3_con_barrio_oficial(test)

if control.get("impLatloncL3", True):
    # Donde tengo l3 pero no latlon, completo con la media de lat lon de ese l3
    train = relleno_latlon_con_media_barrio(train, by="l3")
    test = relleno_latlon_con_media_barrio(test, by="l3")


if control.get("barrioMej", True) and control.get("barrioOK", True):
    train = creo_zonas_mas_precisas(train, uso_osm=control.get("osm", False))
    test = creo_zonas_mas_precisas(test, uso_osm=control.get("osm", False))

# figura_barrios_geo(train, tipo="train", by="l3", show=False)

# %%
###### SEPARO POR TIPO DE PROPIEDAD #######
if control.get("ph2dpto", False):
    # convierto Ph a dpto
    train = convierto_ph(train, debug=False)
    test = convierto_ph(test, debug=False)

if control.get("filDpto", True):
    # Filtro por dpto
    dpto_train = filtro_por_propiedad(
        train, property_type="Departamento", busco_en_titulo=False
    )
    dpto_test = filtro_por_propiedad(test, property_type="Departamento", busco_en_titulo=False)

if control.get("tr", True):
    # Aplico transformaciones
    dpto_train = aplico_transformaciones(dpto_train)
    dpto_test = aplico_transformaciones(dpto_test)

# %%
####### AMBIENTES #######
if control.get("mal2nan_amb", True):
    # Pongo Nans donde no tiene sentido el valor
    dpto_train = mal2nan_ambientes_dpto(dpto_train, debug=False)
    dpto_test = mal2nan_ambientes_dpto(dpto_test, debug=False)

if control.get("impAmbs", True):
    # Imputo los nans
    dpto_train = imputo_ambientes_dpto(
        dpto_train, tipo="train", debug=False, imputo=control.get("ams2imp")
    )  # con rooms solo funciona mejor
    dpto_test = imputo_ambientes_dpto(
        dpto_test, tipo="test", debug=False, imputo=control.get("ams2imp")
    )

####### SUPERFICIE #######
if control.get("mal2nan_sfc", True):
    # Pongo Nans donde no tiene sentido el valor
    dpto_train = mal2nan_sfc_dpto(dpto_train, debug=False)
    dpto_test = mal2nan_sfc_dpto(dpto_test, debug=False)

if control.get("impSfc", True):
    # Imputo los nans
    dpto_train = imputo_sfc_dpto(
        dpto_train, tipo="train", debug=False, imputo=control.get("sfc2imp")
    )
    dpto_test = imputo_sfc_dpto(
        dpto_test, tipo="test", debug=False, imputo=control.get("sfc2imp")
    )

if control.get("sfcMalDec", True):
    # Pongo Nans donde no tiene sentido el valor
    dpto_train = corrijo_superficies_dpto(dpto_train, df_test=dpto_test, debug=False, inflado=True)

####### PRECIO #######
if control.get("mal2nan_precio", True):
    # Pongo Nans donde no tiene sentido el valor
    dpto_train = mal2nan_precio_dpto(dpto_train, debug=False)
    dpto_test = mal2nan_precio_dpto(dpto_test, debug=False)

if control.get("precio_xbxmxp", True):
    # Calculo precio por m2
    dpto_train = precio_xmxbxp(
        dpto_train,
        dpto_test,
        by="l3",
        sub=control.get("osm", False),
        tipo="train",
        sup="surface_covered",
        debug=False,
    )
    dpto_test = precio_xmxbxp(
        dpto_train,
        dpto_test,
        by="l3",
        sub=False,
        sup="surface_covered",
        tipo="test",
        debug=False,
    )

    if control.get("impPrecio", True):
        # Imputo los nans
        dpto_train = imputo_precio(dpto_train, sup="surface_covered")

if control.get("dropOut", True) and control.get("precio_xbxmxp", True):
    # Tiro outliers
    dpto_train = tiro_outliers_dpto(
        dpto_train, 
        df_test=dpto_test,
        minmaxs=control.get("minmax"),
        columnas="all",
        otros=control.get("otros"),
        debug=False)

if control.get("precio_estimado", True):
    # Calculo precio estimado
    dpto_train = precio_estimado(dpto_train, sup="surface_covered")
    dpto_test = precio_estimado(dpto_test, sup="surface_covered")

if control.get("dums", True):
    # Creo dummies
    dpto_train = aplico_dummies_dpto(dpto_train)
    dpto_test = aplico_dummies_dpto(dpto_test)

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
    "precio_xmxbxp",
    "precio_xmxbxp_std",
    "precio_estimado",
    "a_refaccionar",
    "a_nuevo",
    "duplex",
    "lujos",
    "es_PuertoMadero",
]

predictores = list(
    set(predictores_potenciales)
    & set(list(dpto_train.columns))
    & set(list(dpto_test.columns))
)
target = "price"

while True:

    train = dpto_train[predictores + [target]]


    X = train.dropna(subset=predictores + [target])[predictores]
    y = train.dropna(subset=predictores + [target])[target]

    ####### ENTRENO #######
    reg, y_pred, score_test, (X_test, y_test) = entreno_RandomForestRegressor(
        X=X,
        y=y,
        test_size=round(dpto_test.shape[0]/dpto_train.shape[0], 2),
        random_state=42,
        model_params=model_params_default,
    )

    importancia = feature_importance(reg, predictores)
    errores = error_analisis(X_test, y_test, y_pred, by="error")

    features_al_pedo = list(importancia[importancia["importance"] < 2e-4]["feature"])
    if len(features_al_pedo) == 0:
        break   
    predictores = list(set(predictores) - set(features_al_pedo))


# %%
### VOY CON TEST

reg, y_pred, X_test = dalewacho(
    train=dpto_train,
    test=dpto_test,
    predictores=predictores,
    target=target,
    random_state=42,
    model_params=model_params_default,
)

# Comparo con Facu
fac = pd.read_csv(
    data_dir.joinpath("predicciones_facu.csv"),
    index_col="id",
)

fac = fac.loc[X_test.index]

fac["prediccion"] = y_pred

fac["error"] = fac["price"] - fac["prediccion"]
fac["error_abs"] = abs(fac["error"])
fac["error_rel"] = fac["error"] / fac["price"]
rmse = root_mean_squared_error(fac["price"], fac["prediccion"])
print(f"RMSE: {rmse}")
# %%

now = datetime.now().strftime("%Y%m%d-%H%M")

soluciones_path = Path(__file__).parent.joinpath("soluciones")
soluciones_path.mkdir(exist_ok=True)

conf_sol_dpto = "config_soluciones_dpto.csv"
# hago dos archivos uno con la solucion en si y otro con la configuración
if soluciones_path.joinpath(conf_sol_dpto).exists():
    config_soluciones_dpto = pd.read_csv(soluciones_path.joinpath(conf_sol_dpto))
else:
    config_soluciones_dpto = pd.DataFrame(
        {
            "fecha": [],
            "modelo": [],
            "predictores": [],
            "nombre_prueba": [],
            "score_train": [],
            "rmse_sol": [],
        }
    )
config_soluciones_dpto = pd.concat(
    [
        config_soluciones_dpto,
        pd.DataFrame(
            {
                "fecha": [now],
                "modelo": [model_params_default],
                "config": [control],
                "predictores": [predictores],
                "nombre_prueba": [nombre_prueba],
                "score_train": [score_test],
                "rmse_sol": [rmse],
            }
        )
    ]
)

print(config_soluciones_dpto[["score_train", "rmse_sol"]].tail())
config_soluciones_dpto.to_csv(soluciones_path.joinpath(conf_sol_dpto), index=False)
sol_dpto = pd.DataFrame(index=X_test.index, columns={"price": y_pred})
sol_dpto.to_csv(soluciones_path.joinpath(f"sol_dpto_{now}.csv"))

print("asd")
