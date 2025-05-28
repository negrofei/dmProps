import re
import pandas as pd

def es_efectivamente_casa(df: pd.DataFrame):
    df_out = df.copy()
    regex = r"\bcasa\b|\bph\b|\bchalet\b"
    es_casa = df_out["title"].str.contains(
        regex, flags=re.IGNORECASE, regex=True
    ) | df_out["description"].str.contains(regex, flags=re.IGNORECASE, regex=True)
    return es_casa



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
