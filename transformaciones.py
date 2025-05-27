import pandas as pd
import numpy as np


def convierto_pesos_a_dolares(df, debug=True):
    pati = "/home/mfeijoo/Documents/yo/master/dm/fcen-dm-2025-prediccion-precio-de-propiedades/tipos-de-cambio-historicos.csv"
    dolar = pd.read_csv(pati)
    df_out = df.copy()
    df_out = pd.merge(
        df_out,
        dolar[["indice_tiempo", "dolar_estadounidense"]],
        left_on="start_date",
        right_on="indice_tiempo",
    )
    df_out.index = df.index
    en_pesos = df_out["currency"].isin(["ARS"])
    df_out["en_pesos"] = en_pesos
    if debug:
        print(f"Tengo {len(en_pesos)} propiedades en pesos")

    precio_dolares = (
        df_out[en_pesos]["price"] / df_out[en_pesos]["dolar_estadounidense"]
    )

    df_out.loc[en_pesos, "price"] = precio_dolares
    df_out.loc[en_pesos, "currency"] = "USD"
    return df_out.drop(columns=["dolar_estadounidense", "currency", "indice_tiempo"])


def convierto_ph(df, convierto_a="Casa", debug=True):
    df_out = df.copy()
    df_out["es_PH"] = False
    phs = df_out["property_type"] == "PH"
    if debug:
        print(f"Tengo {sum(phs)} PHs")
    if convierto_a not in ["Casa", "Departamento"]:
        raise ValueError(f"Mal convierto_a")
    df_out.loc[phs, "property_type"] = convierto_a
    df_out.loc[phs, "es_PH"] = True
    return df_out


# @title transformo_props
def transformo_props(df: pd.DataFrame):
    df_out = df.copy()

    dummies = pd.get_dummies(df_out["property_type"], prefix="es")
    df_out = pd.concat([df_out, dummies], axis=1)
    return df_out


# @title sup_x_room
def sup_x_room(df: pd.DataFrame):
    df_out = df.copy()
    df_out["sup_x_room"] = df_out["surface_covered"] / df_out["rooms"]
    return df_out


# @title bed-room
def room_menos_bed(df: pd.DataFrame):
    df_out = df.copy()
    df_out["room-bed"] = df_out["rooms"] - df_out["bedrooms"]
    return df_out


# @title total_cov
def total_cov(df: pd.DataFrame):
    df_out = df.copy()
    df_out["total-cov"] = df_out["surface_total"] - df_out["surface_covered"]
    return df_out


# @title ratio_sfc
def ratio_sfc(df: pd.DataFrame):
    df_out = df.copy()
    df_out["ratio_sfc"] = df_out["surface_total"] / df_out["surface_covered"]
    return df_out


# @title sfc_x_bath
def sfc_x_bath(df: pd.DataFrame):
    df_out = df.copy()
    df_out["sfc_x_bath"] = df_out["surface_total"] / df_out["bathrooms"]
    return df_out


# @title binarizo_sups_cuartiles
def binarizo_sups_cuartiles(
    df: pd.DataFrame, cuantiles=[0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
):
    df_out = df.copy()
    if "categoria" in df_out.columns:
        df_out = df_out.drop(columns=["categoria"])
    df_out["categoria"] = pd.qcut(
        df_out["surface_covered"], q=cuantiles, labels=False, duplicates="drop"
    )
    return df_out


# @title aplico_transformaciones
def aplico_transformaciones(df: pd.DataFrame):
    df_out = df.copy()
    df_out = binarizo_sups_cuartiles(df_out)
    df_out = room_menos_bed(total_cov(sup_x_room(sfc_x_bath(df_out))))
    return df_out


def precio_xmxbxp(
    df_train: pd.DataFrame, df_test: pd.DataFrame, by="l3", sub=True, tipo="train", debug=True
):

    df_out = df_train.copy()
    df_out["precio_m2"] = df_out["price"] / df_out["surface_covered"]

    # Mean price per m2 by l3 and l4
    mean_l3 = df_out.groupby("l3")["precio_m2"].mean()
    std_l3 = df_out.groupby("l3")["precio_m2"].std()
    if sub:
        mean_l4 = df_out.groupby("l4")["precio_m2"].mean()
        std_l4 = df_out.groupby("l4")["precio_m2"].std()
        count_l4 = df_out.groupby("l4")["precio_m2"].count()

    # Map means and counts to each row
    df_out["mean_l3"] = df_out["l3"].map(mean_l3)
    df_out["std_l3"] = df_out["l3"].map(std_l3)
    if sub:
        df_out["mean_l4"] = df_out["l4"].map(mean_l4)
        df_out["std_l4"] = df_out["l4"].map(std_l4)
        df_out["count_l4"] = df_out["l4"].map(count_l4)

    if sub:
        # Use l4 mean if l4 is not nan and count >= 3, else l3 mean
        df_out["precio_xmxbxp"] = np.where(
            (~df_out["l4"].isna()) & (df_out["count_l4"] >= 3),
            df_out["mean_l4"],
            df_out["mean_l3"]
        )
        df_out["precio_xmxbxp_std"] = np.where(
            (~df_out["l4"].isna()) & (df_out["count_l4"] >= 3),
            df_out["std_l4"],
            df_out["std_l3"]
        )
    else:
        df_out["precio_xmxbxp"] = df_out["mean_l3"]
        df_out["precio_xmxbxp_std"] = df_out["std_l3"]

    df_out = df_out.drop(columns=["mean_l3", "std_l3"])
    if sub:
        df_out = df_out.drop(columns=["mean_l4", "std_l4", "count_l4"])


    if tipo == "test":
        barrios_precios = df_out[
            [by, "precio_xmxbxp", "precio_xmxbxp_std"]
        ].dropna()
        barrios_precios = barrios_precios.drop_duplicates(subset=[by])

        test_out = df_test.copy()
        if "precio_xmxbxp" in test_out.columns:
            test_out = test_out.drop(columns="precio_xmxbxp")
        if "precio_xmxbxp_std" in test_out.columns:
            test_out = test_out.drop(columns="precio_xmxbxp_std")

        test_out_index = test_out.index
        test_out = test_out.merge(
            barrios_precios, on=[by], how="left"
        ).set_index(test_out_index)
        return test_out

    if debug:
        print(df_out.groupby(by)["precio_m2"].describe())

    return df_out
