import pandas as pd
import numpy as np

from transformaciones import aplico_transformaciones
from texto import rooms_from_text, bedrooms_from_text


# @title correcciones de pedo
def correcciones_de_pedo_train(df: pd.DataFrame):
    df_out = df.copy()
    # Este es dpto
    for index in [15974, 452208, 452209]:
        df_out.loc[index, "property_type"] = "Departamento"

    # Este es dpto
    for index in [413623, 31798]:
        df_out.loc[index, "property_type"] = "Cochera"

    # Tiro estas falopas
    for index in [
        845561,
        364278,
        364296,
        583675,
        857027,
        221666,
        32371,
        32982,
        206820,
        346301,
        617186,
        364277,
        994027,
        49928,
        111953,
        758093,
        88491,
        15038,
        846439,
        118250,
    ]:
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
    df_out.loc[
        [34447, 43990, 63616, 173868, 191965, 191966, 191967, 233270, 315338], "rooms"
    ] = [3, 2, 2, 2, 2, 2, 2, 2, 2]
    df_out.loc[744025, ["lat", "lon"]] = np.nan
    df_out.loc[[381755, 172269], "l3"] = "boca"
    return df_out


# @title mal_values_a_nan_CASA
def mal2nan_ambientes_casa(
    df: pd.DataFrame,
    tipo="train",
    debug=True,
    campos=["rooms", "bedrooms", "bathrooms"],
):
    df_int = df.copy()
    print(f"Arranco con {df_int.shape}")

    ### ROOMS ###
    if "rooms" in campos:
        # Si hay alguno que tiene ambiente 1 le pongo un Nan
        sel = df_int["rooms"] < 1
        df_int.loc[sel, "rooms"] = np.nan
        if debug:
            print(f"Rooms < 1:\t {sum(sel)}")

    ### BEDROOMS ###
    if "bedrooms" in campos:
        # Si hay alguno que tiene bedroom 0 le pongo un Nan
        sel = df_int["bedrooms"] < 0
        df_int.loc[sel, "bedrooms"] = np.nan
        if debug:
            print(f"bedrooms < 0:\t {sum(sel)}")

    ### BATHROOMS ###
    if "bathrooms" in campos:
        # Si hay alguno que tiene bedroom 0 le pongo un Nan
        sel = df_int["bathrooms"] <= 0
        df_int.loc[sel, "bathrooms"] = np.nan
        if debug:
            print(f"bathrooms <= 0:\t {sum(sel)}")
    df_int = aplico_transformaciones(df_int)
    return df_int


def mal2nan_sfc_casa(
    df: pd.DataFrame,
    tipo="train",
    debug=True,
    campos=["surface_total", "surface_covered"],
):
    df_out = df.copy()
    ### SUPERFICIES ###
    if "surface_total" in campos:
        # Si alguno tiene superficie < 10 y no es cochera le pongo un nan
        sel = df_out["surface_total"] < 10
        df_out.loc[sel, "surface_total"] = np.nan
        if debug:
            print(f"surface_total < 10:\t {sum(sel)}")
    if "surface_covered" in campos:
        # Si alguno tiene superficie < 10 y no es cochera le pongo un nan
        sel = df_out["surface_covered"] < 10
        df_out.loc[sel, "surface_covered"] = np.nan
        if debug:
            print(f"surface_covered < 10:\t {sum(sel)}")

    df_out = aplico_transformaciones(df_out)
    return df_out


def mal2nan_precio_casa(df: pd.DataFrame, tipo="train", debug=True):
    df_out = df.copy()
    ### PRECIOS ###
    # si alguno tiene precio = 0 o = 1 o 111111 o alguna boludez así, va Nan
    repetidos = df_out["price"].astype(str).str.match(r"^([0-8])\1+\.\d{1}$")
    ascendentes = df_out["price"].astype(str).str.match(r"^123456789?\.\d{1}$")
    # si alguno tiene precio < 1000
    bajos = df_out["price"] < 1000 * 10  # (seria 1000 dolares x m2 x 10 metros)
    if debug:
        print(f"precios falopa:\t {sum(repetidos) + sum(ascendentes) + sum(bajos)}")
    df_out.loc[repetidos | ascendentes | bajos, "price"] = np.nan

    df_out = aplico_transformaciones(df_out)
    return df_out


# @title tiro_muchos_nans
def tiro_muchos_nans(
    df, debug=True, tol=2, columnas_relevantes=["rooms", "bedrooms", "surface_covered"]
):
    df_out = df.copy()
    tmp = df_out.loc[:, columnas_relevantes]
    falopa = tmp[tmp.isna().sum(axis=1) > tol]
    if debug:
        print(
            f"Tiro las props donde las columnas {columnas_relevantes} tengan más de {tol} NaNs"
        )
        print(df_out.shape)
        print(f"Tengo {len(falopa)} datos falopa")
        print(falopa[columnas_relevantes].head())
    df_out = df_out.drop(index=falopa.index)
    if debug:
        print(df_out.shape)
    return df_out


# @title imputo_ambientes_casa
def imputo_ambientes_casa(
    df: pd.DataFrame,
    tipo="train",
    debug=True,
    imputo=["rooms", "bedrooms", "bathrooms"],
):

    df_out = df.copy()
    if debug:
        print(f"Arranco con {df_out.shape}")

    #### ROOMS ####
    if "rooms" in imputo:
        # Armo flag
        df_out["corrijo_rooms"] = df_out["rooms"].isna()

        # Trato de sacarla del texto
        df_out = rooms_from_text(df_out)

        df_out["rooms"] = df_out["rooms"].fillna(
            df_out[["rooms_from_title", "rooms_from_description"]].max(axis=1)
        )

        # Si correjí rooms y rooms < bedrooms le pongo nan
        df_out.loc[
            (df_out["corrijo_rooms"]) & (df_out["rooms"] <= df_out["bedrooms"]), "rooms"
        ] = np.nan

        # Los nans que quedan lo saco de bedrooms + 1
        df_out["rooms"] = df_out["rooms"].fillna(df_out["bedrooms"] + 1)

        # Y si no el resto lo tiro a la bosta
        if tipo == "train":
            df_out = df_out.dropna(subset=["rooms"])

        # Tiro las columnas de ayuda
        df_out = df_out.drop(columns=["rooms_from_title", "rooms_from_description"])

    if "bedrooms" in imputo:
        #### BEDROOMS ####
        # Armo flag
        df_out["corrijo_bedrooms"] = df_out["bedrooms"].isna()

        # Trato de sacarla del texto
        df_out = bedrooms_from_text(df_out)
        df_out["bedrooms"] = df_out["bedrooms"].fillna(
            df_out[["bedrooms_from_title", "bedrooms_from_description"]].max(axis=1)
        )
        # Si correjí rooms y rooms < bedrooms le pongo nan
        df_out.loc[
            (df_out["corrijo_bedrooms"]) & (df_out["rooms"] <= df_out["bedrooms"]),
            "bedrooms",
        ] = np.nan

        # Imputo con rooms - 1
        df_out["bedrooms"] = df_out["bedrooms"].fillna(df_out["rooms"] - 1)

        # Y si no el resto lo tiro a la bosta
        if tipo == "train":
            df_out = df_out.dropna(subset=["bedrooms"])

    if "bathrooms" in imputo:
        #### BATHROOMS ####
        moda_bathrooms = df_out.groupby(["rooms", "bedrooms"])["bathrooms"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        # Armo flag
        df_out["corrijo_bathrooms"] = df_out["bathrooms"].isna()

        df_out["bathrooms_mode"] = df_out.set_index(["rooms", "bedrooms"]).index.map(
            moda_bathrooms
        )

        df_out["bathrooms"] = df_out["bathrooms"].fillna(df_out["bathrooms_mode"])

        df_out = df_out.drop(columns=["bathrooms_mode"])

        # Lo que resta le pongo 1
        df_out["bathrooms"] = df_out["bathrooms"].replace(np.nan, 1)

    df_out = aplico_transformaciones(df_out)
    return df_out


def imputo_sfc_casa(
    df: pd.DataFrame,
    tipo="train",
    debug=True,
    imputo=["surface_total", "surface_covered", "invierto_surface"],
):
    """Hago lo que hice en la entrega bien, después vemos"""
    df_out = df.copy()
    if debug:
        print(f"Arranco con {df_out.shape}")

    if "surface_total" in imputo:
        # Si no tengo superficie total tomo la superficie covered
        if debug:
            nans = sum(df_out["surface_total"].isna())
            print(f"Tengo {nans} surface_total nan")
        df_out["imputo_surface_total"] = df_out["surface_total"].isna()
        df_out["surface_total"] = df_out["surface_total"].fillna(
            df_out["surface_covered"]
        )
        # si no tengo surface covered tomo la media segun room
        df_out["surface_total"] = df_out["surface_total"].fillna(
            df_out.groupby("rooms")["surface_total"].transform("mean")
        )

    if "surface_covered" in imputo:
        # Si no tengo surface_covered tomo la surface_total
        if debug:
            nans = sum(df_out["surface_covered"].isna())
            print(f"Tengo {nans} surface_covered nan")
        df_out["imputo_surface_covered"] = df_out["surface_covered"].isna()
        df_out["surface_covered"] = df_out["surface_covered"].fillna(
            df_out["surface_total"]
        )
        # si no tengo surface_total tomo la media segun room
        df_out["surface_covered"] = df_out["surface_covered"].fillna(
            df_out.groupby("rooms")["surface_covered"].transform("mean")
        )

    # Si no tengo ninguna los tiro a la mierda
    if tipo == "train":
        if debug:
            nans = (
                (df_out["surface_covered"].isna()) & (df_out["surface_total"].isna())
            ).sum()
            print(f"Tengo {nans} surface_covered y surface_total nan")
        df_out = df_out.dropna(subset=["surface_total", "surface_covered"])

    if "invierto_surface" in imputo:
        # Si tengo surface_covered > surface_total, las invierto
        # invierto las superficies que estén al reves
        df_out["inverti_sups"] = False
        invertidos = df_out["surface_total"] < df_out["surface_covered"]
        mals = df_out[invertidos]
        if debug:
            print(f"Tengo {len(mals)} superficies invertidas")
            # print(mals[["rooms", "surface_total", "surface_covered", "total-cov"]].head())

        for idx, row in mals.iterrows():
            df_out.loc[idx, ["surface_total", "surface_covered"]] = df_out.loc[
                idx, ["surface_covered", "surface_total"]
            ].values
            df_out.at[idx, "inverti_sups"] = True
    print(f"Termino con {df_out.shape}")
    return df_out


def imputo_precio(df: pd.DataFrame, sup="surface_covered"):
    df_out = df.copy()
    # Si el precio es NaN lo imputo con surface_covered * precio_x_m2_x_barrio_x_prop
    df_out["price"] = df_out["price"].fillna(df_out[sup] * df_out["precio_xmxbxp"])
    return df_out


# @title mal_values_a_nan_dpto
def mal2nan_ambientes_dpto(
    df: pd.DataFrame,
    tipo="train",
    debug=True,
    campos=["rooms", "bedrooms", "bathrooms"],
):
    df_int = df.copy()
    print(f"Arranco con {df_int.shape}")

    ### ROOMS ###
    if "rooms" in campos:
        # Si hay alguno que tiene ambiente 1 le pongo un Nan
        sel = df_int["rooms"] < 1
        df_int.loc[sel, "rooms"] = np.nan
        if debug:
            print(f"Rooms < 1:\t {sum(sel)}")

    ### BEDROOMS ###
    if "bedrooms" in campos:
        # Si hay alguno que tiene bedroom 0 le pongo un Nan
        sel = df_int["bedrooms"] < 0
        df_int.loc[sel, "bedrooms"] = np.nan
        if debug:
            print(f"bedrooms < 0:\t {sum(sel)}")

    ### BATHROOMS ###
    if "bathrooms" in campos:
        # Si hay alguno que tiene bedroom 0 le pongo un Nan
        sel = df_int["bathrooms"] < 1
        df_int.loc[sel, "bathrooms"] = np.nan
        if debug:
            print(f"bathrooms <= 0:\t {sum(sel)}")
    df_int = aplico_transformaciones(df_int)
    return df_int


def mal2nan_sfc_dpto(
    df: pd.DataFrame,
    tipo="train",
    debug=True,
    campos=["surface_total", "surface_covered"],
):
    df_out = df.copy()
    ### SUPERFICIES ###
    if "surface_total" in campos:
        # Si alguno tiene superficie < 10 y no es cochera le pongo un nan
        sel = df_out["surface_total"] < 10
        df_out.loc[sel, "surface_total"] = np.nan
        if debug:
            print(f"surface_total < 10:\t {sum(sel)}")
    if "surface_covered" in campos:
        # Si alguno tiene superficie < 10 y no es cochera le pongo un nan
        sel = df_out["surface_covered"] < 10
        df_out.loc[sel, "surface_covered"] = np.nan
        if debug:
            print(f"surface_covered < 10:\t {sum(sel)}")

    df_out = aplico_transformaciones(df_out)
    return df_out


def mal2nan_precio_dpto(df: pd.DataFrame, tipo="train", debug=True):
    df_out = df.copy()
    ### PRECIOS ###
    # si alguno tiene precio = 0 o = 1 o 111111 o alguna boludez así, va Nan
    repetidos = df_out["price"].astype(str).str.match(r"^([0-8])\1+\.\d{1}$")
    ascendentes = df_out["price"].astype(str).str.match(r"^123456789?\.\d{1}$")
    # si alguno tiene precio < 1000
    bajos = df_out["price"] < 1000 * 10  # (seria 1000 dolares x m2 x 10 metros)
    if debug:
        print(f"precios falopa:\t {sum(repetidos) + sum(ascendentes) + sum(bajos)}")
    df_out.loc[repetidos | ascendentes | bajos, "price"] = np.nan

    df_out = aplico_transformaciones(df_out)
    return df_out


# @title imputo_ambientes_dpto
def imputo_ambientes_dpto(
    df: pd.DataFrame,
    tipo="train",
    debug=True,
    imputo=["rooms", "bedrooms", "bathrooms"],
):

    df_out = df.copy()
    if debug:
        print(f"Arranco con {df_out.shape}")

    #### ROOMS ####
    if "rooms" in imputo:
        # Armo flag
        df_out["corrijo_rooms"] = df_out["rooms"].isna()

        # Trato de sacarla del texto
        df_out = rooms_from_text(df_out)

        df_out["rooms"] = df_out["rooms"].fillna(
            df_out[["rooms_from_title", "rooms_from_description"]].max(axis=1)
        )

        # Si correjí rooms y rooms < bedrooms le pongo nan
        df_out.loc[
            (df_out["corrijo_rooms"]) & (df_out["rooms"] <= df_out["bedrooms"]), "rooms"
        ] = np.nan

        # Los nans que quedan lo saco de bedrooms + 1
        df_out["rooms"] = df_out["rooms"].fillna(df_out["bedrooms"] + 1)

        # Y si no el resto lo tiro a la bosta
        if tipo == "train":
            df_out = df_out.dropna(subset=["rooms"])

        # Tiro las columnas de ayuda
        df_out = df_out.drop(columns=["rooms_from_title", "rooms_from_description"])

    if "bedrooms" in imputo:
        #### BEDROOMS ####
        # Armo flag
        df_out["corrijo_bedrooms"] = df_out["bedrooms"].isna()

        # Trato de sacarla del texto
        df_out = bedrooms_from_text(df_out)
        df_out["bedrooms"] = df_out["bedrooms"].fillna(
            df_out[["bedrooms_from_title", "bedrooms_from_description"]].max(axis=1)
        )
        # Tiro las columnas de ayuda
        df_out = df_out.drop(columns=["bedrooms_from_title", "bedrooms_from_description"])
        # Si correjí rooms y rooms < bedrooms le pongo nan
        df_out.loc[
            (df_out["corrijo_bedrooms"]) & (df_out["rooms"] <= df_out["bedrooms"]),
            "bedrooms",
        ] = np.nan

        # Imputo con rooms - 1
        df_out["bedrooms"] = df_out["bedrooms"].fillna(df_out["rooms"] - 1)

        # si tengo un monoambiente, mando bedrooms = 0
        df_out.loc[df_out["rooms"] == 1, "bedrooms"] = 0
        # Y si no el resto lo tiro a la bosta
        if tipo == "train":
            df_out = df_out.dropna(subset=["bedrooms"])

    if "bathrooms" in imputo:
        #### BATHROOMS ####
        moda_bathrooms = df_out.groupby(["rooms", "bedrooms"])["bathrooms"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        # Armo flag
        df_out["corrijo_bathrooms"] = df_out["bathrooms"].isna()

        df_out["bathrooms_mode"] = df_out.set_index(["rooms", "bedrooms"]).index.map(
            moda_bathrooms
        )

        df_out["bathrooms"] = df_out["bathrooms"].fillna(df_out["bathrooms_mode"])

        df_out = df_out.drop(columns=["bathrooms_mode"])

        # Lo que resta le pongo 1
        df_out["bathrooms"] = df_out["bathrooms"].replace(np.nan, 1)

    df_out = aplico_transformaciones(df_out)
    return df_out


def imputo_sfc_dpto(
    df: pd.DataFrame,
    tipo="train",
    debug=True,
    imputo=["surface_total", "surface_covered", "invierto_surface"],
):
    """Hago lo que hice en la entrega bien, después vemos"""
    df_out = df.copy()
    if debug:
        print(f"Arranco con {df_out.shape}")

    if "surface_total" in imputo:
        # Si no tengo superficie total tomo la superficie covered
        if debug:
            nans = sum(df_out["surface_total"].isna())
            print(f"Tengo {nans} surface_total nan")
        df_out["imputo_surface_total"] = df_out["surface_total"].isna()
        df_out["surface_total"] = df_out["surface_total"].fillna(
            df_out["surface_covered"]
        )
        # si no tengo surface covered tomo la media segun room
        df_out["surface_total"] = df_out["surface_total"].fillna(
            df_out.groupby("rooms")["surface_total"].transform("mean")
        )

    if "surface_covered" in imputo:
        # Si no tengo surface_covered tomo la surface_total
        if debug:
            nans = sum(df_out["surface_covered"].isna())
            print(f"Tengo {nans} surface_covered nan")
        df_out["imputo_surface_covered"] = df_out["surface_covered"].isna()
        df_out["surface_covered"] = df_out["surface_covered"].fillna(
            df_out["surface_total"]
        )
        # si no tengo surface covered tomo la media segun room
        df_out["surface_covered"] = df_out["surface_covered"].fillna(
            df_out.groupby("rooms")["surface_covered"].transform("mean")
        )

    # Si no tengo ninguna los tiro a la mierda
    if tipo == "train":
        if debug:
            nans = (
                (df_out["surface_covered"].isna()) & (df_out["surface_total"].isna())
            ).sum()
            print(f"Tengo {nans} surface_covered y surface_total nan")
        df_out = df_out.dropna(subset=["surface_total", "surface_covered"])

    if "invierto_surface" in imputo:
        # Si tengo surface_covered > surface_total, las invierto
        # invierto las superficies que estén al reves
        df_out["inverti_sups"] = False
        invertidos = df_out["surface_total"] < df_out["surface_covered"]
        mals = df_out[invertidos]
        if debug:
            print(f"Tengo {len(mals)} superficies invertidas")
            # print(mals[["rooms", "surface_total", "surface_covered", "total-cov"]].head())

        for idx, row in mals.iterrows():
            df_out.loc[idx, ["surface_total", "surface_covered"]] = df_out.loc[
                idx, ["surface_covered", "surface_total"]
            ].values
            df_out.at[idx, "inverti_sups"] = True
    print(f"Termino con {df_out.shape}")
    return df_out


#@title corrijo_surface_total_y_covered_inflado
def corrijo_surface_total_y_covered_inflado(df, indexes, tol=0.5, debug=False, low=0.025, upp=0.975):
    df_out = df.copy()
    factores = [1000, 100, 10]  # mayor a menor
    df_out["surface_totalycov_inflado"] = False

    # Calculo cuantiles por grupo
    quantiles = df_out.groupby(["rooms", "bedrooms"])[["surface_total", "surface_covered"]].quantile([low, 0.5, upp])

    for idx in indexes:
        if debug:
            print(f"\n{idx=}")
        try:
            row = df_out.loc[idx]
            r, b = row["rooms"], row["bedrooms"]
            sup_total, sup_cov = row["surface_total"], row["surface_covered"]

            if pd.notna(sup_total) and pd.notna(sup_cov) and sup_cov > 0:
                for f in factores:
                    corregido_total = False
                    corregido_covered = False

                    st_corr = sup_total / f
                    sc_corr = sup_cov / f

                    # Si tenemos cuantiles para este grupo
                    if (r, b) in quantiles.index:
                        st_low = quantiles["surface_total"].get((r, b, low), np.nan)
                        st_upp = quantiles["surface_total"].get((r, b, upp), np.nan)

                        if debug:
                            print(f"idx: {idx}, factor: {f}, st_corr: {st_corr}, low{low}-upp{upp}: {st_low}-{st_upp}")

                        # Verifico que el valor corregido caiga dentro del rango razonable
                        if pd.notna(st_low) and pd.notna(st_upp) and st_low * (1 - tol) < st_corr < st_upp * (1 + tol):
                            df_out.loc[idx, "surface_total"] = st_corr
                            df_out.loc[idx, "surface_totalycov_inflado"] = True
                            corregido_total = True
                            if debug:
                                print(f"surface_total corregido {st_corr}")

                        st_low = quantiles["surface_covered"].get((r, b, low), np.nan)
                        st_upp = quantiles["surface_covered"].get((r, b, upp), np.nan)

                        if debug:
                            print(f"idx: {idx}, factor: {f}, sc_corr: {sc_corr}, low {low} - upp {upp}: {st_low}-{st_upp}")

                        # Verifico que el valor corregido caiga dentro del rango razonable
                        if pd.notna(st_low) and pd.notna(st_upp) and st_low * (1 - tol) < sc_corr < st_upp * (1 + tol):
                            df_out.loc[idx, "surface_covered"] = sc_corr
                            df_out.loc[idx, "surface_totalycov_inflado"] = True
                            corregido_covered = True
                            if debug:
                                print(f"surface_covered corregido {sc_corr}")
                    # Si alguno de los dos fue corregido, no pruebo con más factores
                    if corregido_total or corregido_covered:
                        break
        except KeyError:
            if debug:
                print(f"Índice {idx} no está en el DataFrame. Lo salto.")
            continue

    return df_out

#@title corrijo_surface_inflado
def corrijo_surface_inflado(df, indexes, tol=0.5, surface="surface_total", debug=False):
    df_out = df.copy()
    factores = [1000, 100, 10]  # mayor a menor
    df_out[f"{surface}_inflado"] = False  # inicializa como False

    for idx in indexes:
        try:
            sup_total = df_out.loc[idx, "surface_total"]
            sup_cov = df_out.loc[idx, "surface_covered"]
            if surface == "surface_total":
                sup_mal = sup_total
            elif surface == "surface_covered":
                sup_mal = sup_cov
            else:
                raise ValueError(f"mal surface")
            if pd.notna(sup_total) and pd.notna(sup_cov) and sup_cov > 0:
                if surface == "surface_total":
                    ratio = sup_total / sup_cov
                else:
                    ratio = sup_cov /sup_total
                for f in factores:
                    if debug:
                        print(f"{idx=} - {f=} - {sup_total=:.1f} - {sup_cov=:.1f} - {ratio=:.1f}")
                        print(f"(1-tol) * f < ratio < (1+tol) * f: {(1-tol)*f} < {ratio} < {(1+tol)*f}")
                    condicion1 = (1-tol) * f < ratio < (1+tol) * f
                    if surface == "surface_total":
                        condicion2 = sup_mal/f > sup_cov
                    else:
                        condicion2 = sup_mal/f < sup_total
                    if condicion1 and condicion2 :
                        if debug:
                            print(f"{surface} corregida {sup_mal/f}\n")
                        df_out.loc[idx, surface] = sup_mal / f
                        df_out.loc[idx, f"{surface}_inflado"] = True
                        break
        except KeyError:
            print(f"Índice {idx} no está en el DataFrame. Lo salto.")
            continue

    return df_out


#@title corrijo_superficies_dpto
def corrijo_superficies_dpto(df: pd.DataFrame, df_test: pd.DataFrame, debug=True, inflado=True):
    df_out = df.copy()

    # aplico una funcion para corregir unos surface_totla y surface_cov cuando los dos se van a la mierda
    # malitos = (df_out["sup_x_room"] > df_bueno["sup_x_room"].quantile(1)) | (df_out["surface_total"]>2500) | (abs(df_out["total-cov"]) > df_out["surface_covered"] ) # ya esto es una guasada
    mucha_total = (df_out["surface_total"]>df_test["surface_total"].quantile(1))
    mucha_covered = (df_out["surface_covered"]>df_test["surface_covered"].quantile(1))
    if inflado:
        indices = df_out.loc[mucha_total & mucha_covered].index
        if debug:
            print(f"Tengo {sum(mucha_total)} con sup > al max de test")
            print(f"Tengo {sum(mucha_covered)} con cov > al max de test")
            print(f"Tengo total {len(indices)}")
        df_out = corrijo_surface_total_y_covered_inflado(df_out, indexes=indices, tol=0, debug=debug)
        df_out = aplico_transformaciones(df_out)


    # aplico una funcion para corregir unos surface_total que son cualquiera
    mucha_tot_cov = (df_out["total-cov"]>df_test["total-cov"].quantile(1))
    if inflado:
        if debug:
            print(f"Tengo {sum(mucha_tot_cov)} con sup >>> cov")
        df_out = corrijo_surface_inflado(df_out, surface="surface_total", indexes=df_out.loc[mucha_tot_cov].index, debug=debug)
        df_out = aplico_transformaciones(df_out)

    # aplico una funcion para corregir unos covered que son cualquiera
    mucha_cov_tot = (df_out["total-cov"]< -1*df_test["total-cov"].quantile(1))
    if inflado:
        if debug:
            print(f"Tengo {sum(mucha_cov_tot)} con sup <<< cov")
        df_out = corrijo_surface_inflado(df_out, surface="surface_covered", indexes=df_out.loc[mucha_cov_tot].index, debug=debug)
        df_out = aplico_transformaciones(df_out)

    mucha_cov_tot = (df_out["total-cov"]< -1*df_test["total-cov"].quantile(1))
    # # invierto las superficies que estén al reves
    # df_out["inverti_sups"] = False
    # invertidos = (df_out["surface_total"] < df_out["surface_covered"])
    # mals = df_out[invertidos]
    # if debug:
    #     print(f"Tengo {len(mals)} superficies invertidas")
    #     print(mals[["rooms", "surface_total", "surface_covered", "total-cov"]].head())

    # for idx, row in mals.iterrows():
    #     df_out.loc[idx, ["surface_total", "surface_covered"]] = df_out.loc[idx, ["surface_covered", "surface_total"]].values
    #     df_out.at[idx, "inverti_sups"] = True

    df_out = aplico_transformaciones(df_out)

    return aplico_transformaciones(df_out)

