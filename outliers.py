import pandas as pd
from transformaciones import aplico_transformaciones

def tiro_outliers_casa(df: pd.DataFrame, df_test: pd.DataFrame, columnas=["rooms", "bedrooms", "surface_total"], filtro_cuantil=0.99, debug=True):
    df_out = df.copy()
    if debug:
        print(f"Arranco con {df_out.shape}")
    
    # Defino los rangos de los outliers segñun lo que haya en el test
    minmaxs = {
        "rooms": [1, df_test["rooms"].quantile(filtro_cuantil)],
        "bedrooms": [0, df_test["bedrooms"].quantile(filtro_cuantil)],
        "bathrooms": [1, df_test["bathrooms"].quantile(filtro_cuantil)],
        "surface_total": [10, df_test["surface_total"].quantile(filtro_cuantil)],
        "surface_covered": [10, df_test["surface_covered"].quantile(filtro_cuantil)],
        "sup_x_room": [15, df_test["sup_x_room"].quantile(filtro_cuantil)],
        # "total-cov": [0, df_test["total-cov"].quantile(filtro_cuantil)],
        "room-bed": [1, df_test["room-bed"].quantile(filtro_cuantil)],
        "sfc_x_bath" : [10, df_test["sfc_x_bath"].quantile(filtro_cuantil)],
    }
    if debug:
        print(f"Propiedades que no entran en los rangos establecidos\n")
        for col in columnas:
            print(f"{col}: {len(df_out[(df_out[col]<minmaxs[col][0]) | (df_out[col]>minmaxs[col][1])])}")
    
    for col in columnas:
        df_out = df_out[(df_out[col]>=minmaxs[col][0]) & (df_out[col]<=minmaxs[col][1])]

    df_out = aplico_transformaciones(df_out)
    print(f"Termino con {df_out.shape}")
    return df_out