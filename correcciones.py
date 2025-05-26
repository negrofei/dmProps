import pandas as pd
import numpy as np

from transformaciones import aplico_transformaciones
from texto import rooms_from_text, bedrooms_from_text

#@title correcciones de pedo
def correcciones_de_pedo_train(df: pd.DataFrame):
    df_out = df.copy()
    # Este es dpto
    for index in [15974, 452208, 452209]:
        df_out.loc[index, "property_type"] = "Departamento"

    # Este es dpto
    for index in [413623, 31798]:
        df_out.loc[index, "property_type"] = "Cochera"

    # Tiro estas falopas
    for index in [845561, 364278, 364296,583675, 857027, 221666, 32371, 32982, 206820, 346301, 617186, 364277, 994027, 49928, 111953, 758093, 88491, 15038, 846439,118250]:
        df_out = df_out.drop(index=index)
    
    df_out.loc[14842, "rooms"] = 13
    df_out.loc[616969, "surface_total"] = 114
    df_out.loc[640768, "surface_total"] = 120
    df_out.loc[831195, "surface_total"] = 77
    df_out.loc[88491, "surface_total"] = 126
    df_out.loc[544806, "surface_covered"] = 126
    df_out.loc[969228, "surface_total"] = 183
    df_out.loc[276679, "l3"] = "Belgrano"
    df_out.loc[[184214, 237857], "bedrooms"] = 0
    df_out.loc[72997, ["rooms", "bedrooms"]] = [3, 2]
    df_out.loc[782283, "surface_total"] = 69.1
    df_out.loc[627201, "rooms"] = 6
    df_out.loc[699889, "rooms"] = 4
    return df_out

def correcciones_de_pedo_test(df: pd.DataFrame):
    df_out = df.copy()
    df_out.loc[[34447, 43990, 63616, 173868, 191965, 191966, 191967, 233270, 315338], "rooms"] = [3, 2, 2, 2, 2, 2, 2, 2, 2]
    df_out.loc[744025, ["lat", "lon"]] = np.nan
    df_out.loc[[381755, 172269], "l3"] = "boca"
    return df_out


#@title mal_values_a_nan_CASA
def mal_values_a_nan_CASA(df, tipo="train", debug=True):
    df_int = df.copy()
    print(f"Arranco con {df_int.shape}")

    ### ROOMS ###
    # Si hay alguno que tiene ambiente 1 le pongo un Nan
    sel = (df_int["rooms"]<1)
    df_int.loc[sel, "rooms"] = np.nan
    if debug:
        print(f"Rooms < 1:\t {sum(sel)}")

    ### BEDROOMS ###
    # Si hay alguno que tiene bedroom 0 le pongo un Nan
    sel = (df_int["bedrooms"]<0)
    df_int.loc[sel, "bedrooms"] = np.nan
    if debug:
        print(f"bedrooms < 0:\t {sum(sel)}")


    ### BATHROOMS ###
    # Si hay alguno que tiene bedroom 0 le pongo un Nan
    sel = (df_int["bathrooms"]<=0)
    df_int.loc[sel, "bathrooms"] = np.nan
    if debug:
        print(f"bathrooms <= 0:\t {sum(sel)}")

    ### REALCION BEDROOM ROOM
    sel = (df_int["bedrooms"] > df_int["rooms"])
    if debug:
        print(f"bedrooms > rooms:\t {sum(sel)}")

    ### SUPERFICIES ###
    # Si alguno tiene superficie < 10 y no es cochera le pongo un nan
    sel = (df_int["surface_total"]<10)
    df_int.loc[sel, "surface_total"] = np.nan
    if debug:
        print(f"surface_total < 10:\t {sum(sel)}")
    sel = (df_int["surface_covered"]<10)
    df_int.loc[sel, "surface_covered"] = np.nan
    if debug:
        print(f"surface_covered < 10:\t {sum(sel)}")

    ### RELACION SUP TOT SUP COV
    # Si tengo surface_covered > surface_total
    sel = (df_int["surface_covered"] > df_int["surface_total"])
    if debug:
        print(f"surface_covered > surface_total:\t {sum(sel)}")

    ### PRECIOS ###
    # si alguno tiene precio = 0 o = 1 o 111111 o alguna boludez así, va Nan
    repetidos = df_int["price"].astype(str).str.match(r'^([0-8])\1+\.\d{1}$')
    ascendentes = df_int["price"].astype(str).str.match(r'^123456789?\.\d{1}$')

    if debug:
        print(f"precios falopa:\t {sum(repetidos) + sum(ascendentes)}")
    df_int.loc[repetidos | ascendentes, "price"] = np.nan
    print("\n")

    df_int = aplico_transformaciones(df_int)
    return df_int


#@title tiro_muchos_nans
def tiro_muchos_nans(df, debug=True, tol=2, columnas_relevantes=["rooms", "bedrooms", "surface_covered"]):
    df_out = df.copy()
    tmp = df_out.loc[:, columnas_relevantes]
    falopa = tmp[tmp.isna().sum(axis=1) > tol]
    if debug:
        print(f"Tiro las props donde las columnas {columnas_relevantes} tengan más de {tol} NaNs")
        print(df_out.shape)
        print(f"Tengo {len(falopa)} datos falopa")
        print(falopa[columnas_relevantes].head())
    df_out = df_out.drop(index=falopa.index)
    if debug:
        print(df_out.shape)
    return df_out

#@title imputo_ambientes_casa
def imputo_ambientes_casa(df, tipo="train", debug=True):
    df_out = df.copy()
    if debug:
        print(f"Arranco con {df_out.shape}")

    #### ROOMS ####
    # Armo flag
    df_out["corrijo_rooms"] = df_out["rooms"].isna()

    # Trato de sacarla del texto
    df_out = rooms_from_text(df_out)

    df_out["rooms"] = df_out["rooms"].fillna(
        df_out[["rooms_from_title", "rooms_from_description"]].max(axis=1)
    )
    
    # Si correjí rooms y rooms < bedrooms le pongo nan
    df_out.loc[(df_out["corrijo_rooms"])&(df_out["rooms"] <= df_out["bedrooms"]), "rooms"] = np.nan

    # Los nans que quedan lo saco de bedrooms + 1
    df_out["rooms"] = df_out["rooms"].fillna(df_out["bedrooms"] + 1)

    # Y si no el resto lo tiro a la bosta
    if tipo=="train":
        df_out = df_out.dropna(subset=["rooms"])
    
    # Tiro las columnas de ayuda
    df_out = df_out.drop(columns=["rooms_from_title", "rooms_from_description"])
    
    #### BEDROOMS ####
    # Armo flag
    df_out["corrijo_bedrooms"] = df_out["bedrooms"].isna()

    # Imputo con rooms - 1
    df_out["bedrooms"] = df_out["bedrooms"].fillna(df_out["rooms"] - 1)

    # Y si no el resto lo tiro a la bosta
    if tipo=="train":
        df_out = df_out.dropna(subset=["bedrooms"])


    #### BATHROOMS ####
    moda_bathrooms = (
        df_out
        .groupby(["rooms", "bedrooms"])["bathrooms"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )
    # Armo flag
    df_out["corrijo_bathrooms"] = df_out["bathrooms"].isna()

    df_out["bathrooms_mode"] = df_out.set_index(["rooms", "bedrooms"]).index.map(moda_bathrooms)

    df_out["bathrooms"] = df_out["bathrooms"].fillna(df_out["bathrooms_mode"])

    df_out = df_out.drop(columns=["bathrooms_mode"])

    # Lo que resta le pongo 1 
    df_out["bathrooms"] = df_out["bathrooms"].replace(np.nan, 1)


    df_out = aplico_transformaciones(df_out)
    return df_out

