import pandas as pd
import numpy as np

def convierto_pesos_a_dolares(df, debug=True):
    pati = "/home/mfeijoo/Documents/yo/master/dm/fcen-dm-2025-prediccion-precio-de-propiedades/tipos-de-cambio-historicos.csv"
    dolar = pd.read_csv(pati)
    df_out =  df.copy()
    df_out = pd.merge(df_out, dolar[["indice_tiempo", "dolar_estadounidense"]], left_on="start_date", right_on="indice_tiempo")
    df_out.index = df.index  
    en_pesos = (df_out["currency"].isin(["ARS"]))
    df_out["en_pesos"] = en_pesos
    if debug:
        print(f"Tengo {len(en_pesos)} propiedades en pesos")
    
    precio_dolares = df_out[en_pesos]["price"] / df_out[en_pesos]["dolar_estadounidense"]
    
    df_out.loc[en_pesos, "price"] = precio_dolares
    df_out.loc[en_pesos, "currency"] = "USD"
    return df_out.drop(columns=["dolar_estadounidense", "currency", "indice_tiempo"])


def convierto_ph(df, convierto_a="Casa", debug=True):
    df_out = df.copy()
    df_out["es_PH"] = False
    phs = (df_out["property_type"] == "PH")
    if debug:
        print(f"Tengo {sum(phs)} PHs")
    if convierto_a not in ["Casa", "Departamento"]:
        raise ValueError(f"Mal convierto_a")
    df_out.loc[phs, "property_type"] = convierto_a
    df_out.loc[phs, "es_PH"] == True
    return df_out
