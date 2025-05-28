import pandas as pd
import numpy as np
from transformaciones import aplico_transformaciones, precio_xmxbxp


def tiro_outliers_casa(
    df: pd.DataFrame,
    df_test: pd.DataFrame = None,
    minmaxs: dict = {},
    columnas=["rooms", "bedrooms", "surface_total"],
    otros: dict = {},
    debug=True,
):
    df_out = df.copy()
    if debug:
        print(f"Arranco con {df_out.shape}")

    if debug:
        print(f"Propiedades que no entran en los rangos establecidos\n")
        for col in columnas:
            print(
                f"{col}: {len(df_out[(df_out[col]<minmaxs[col][0]) | (df_out[col]>minmaxs[col][1])])}"
            )
    if columnas == "all":
        columnas = minmaxs.keys()
    for col in columnas:
        df_out = df_out[
            (df_out[col] >= minmaxs[col][0]) & (df_out[col] <= minmaxs[col][1])
        ]

    # Adicional para precio_m2
    sigma = otros.get("sigma", None)
    if sigma:
        df_out = df_out[(df_out["precio_m2"] - df_out["precio_xmxbxp"]).abs() < df_out["precio_xmxbxp_std"]*sigma]
    diff_pm2 = otros.get("diff_pm2", None)
    if diff_pm2:
        df_out = df_out[(df_out["precio_m2"] - df_out["precio_xmxbxp"]).abs() < diff_pm2]

    df_out = aplico_transformaciones(df_out)
    df_out = precio_xmxbxp(
        df_out,
        df_test,
        by="l3",
        sub=False,
        tipo="train",
        sup="surface_covered",
        debug=False,
    )
    print(f"Termino con {df_out.shape}")
    return df_out


def tiro_outliers_dpto(
    df: pd.DataFrame,
    df_test: pd.DataFrame = None,
    minmaxs: dict = {},
    columnas=["rooms", "bedrooms", "surface_total"],
    otros: dict = {},
    debug=True,
):
    df_out = df.copy()
    if debug:
        print(f"Arranco con {df_out.shape}")

    if debug:
        print(f"Propiedades que no entran en los rangos establecidos\n")
        for col in columnas:
            print(
                f"{col}: {len(df_out[(df_out[col]<minmaxs[col][0]) | (df_out[col]>minmaxs[col][1])])}"
            )
    if columnas == "all":
        columnas = minmaxs.keys()
    for col in columnas:
        df_out = df_out[
            (df_out[col] >= minmaxs[col][0]) & (df_out[col] <= minmaxs[col][1])
        ]

    # Adicional para precio_m2
    sigma = otros.get("sigma", None)
    if sigma:
        df_out = df_out[(df_out["precio_m2"] - df_out["precio_xmxbxp"]).abs() < df_out["precio_xmxbxp_std"]*sigma]
    diff_pm2 = otros.get("diff_pm2", None)
    if diff_pm2:
        df_out = df_out[(df_out["precio_m2"] - df_out["precio_xmxbxp"]).abs() < diff_pm2]

    df_out = aplico_transformaciones(df_out)
    df_out = precio_xmxbxp(
        df_out,
        df_test,
        by="l3",
        sub=False,
        tipo="train",
        sup="surface_covered",
        debug=False,
    )
    print(f"Termino con {df_out.shape}")
    return df_out
