import pandas as pd
import spacy
nlp = spacy.load("es_core_news_sm")
import re

# Trato de sacar la info del titulo y la descripción
def rooms_from_text(df: pd.DataFrame):
    df_out = df.copy()
    # Map Spanish words to numbers
    word2num = {
        "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
        "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10
    }
    # Regex: number or word, optional spaces, amb/ambiente(s)
    pattern = re.compile(
        r"(?P<num>\d+|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s*(amb|ambiente?s?)",
        re.IGNORECASE
    )

    def extract_rooms(text):
        if pd.isna(text):
            return None
        match = pattern.search(text)
        if match:
            val = match.group("num").lower()
            if val.isdigit():
                return float(val)
            return float(word2num.get(val, None))
        return None
    
    for campo in ["title", "description"]:
        df_out[f"rooms_from_{campo}"] = df_out[campo].apply(extract_rooms)
        
    return df_out


def bedrooms_from_text(df: pd.DataFrame):
    df_out = df.copy()
    # Map Spanish words to numbers
    word2num = {
        "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
        "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10
    }
    # Regex: number or word, optional spaces, amb/ambiente(s)
    pattern = re.compile(
        r"(?P<num>\d+|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s*(hab(?:itaci[oó]n(?:es)?)?|dormitorio[s]?)",
        re.IGNORECASE
    )

    def extract_rooms(text):
        if pd.isna(text):
            return None
        match = pattern.search(text)
        if match:
            val = match.group("num").lower()
            if val.isdigit():
                return float(val)
            return float(word2num.get(val, None))
        return None
    
    for campo in ["title", "description"]:
        df_out[f"bedrooms_from_{campo}"] = df_out[campo].apply(extract_rooms)
        
    return df_out

#@title a_refaccionar
def a_refaccionar(df: pd.DataFrame):
    df_out  = df.copy()
    # A refaccionar
    keywords = [
        "a refaccionar", "a reciclar", "a remodelar", "a restaurar",
        "oportunidad lote", "a mejorar"
    ]
    regex = "|".join([fr"\b{palabra}\b" for palabra in keywords])
    sel = (
        df_out["description"].fillna("").str.lower().str.contains(regex, na=False) |
        df_out["title"].fillna("").str.lower().str.contains(regex, na=False)
    )
    df_out["a_refaccionar"] = sel
    return df_out

#@title a_nuevo
def a_nuevo(df: pd.DataFrame):
    df_out  = df.copy()
    # A refaccionar
    keywords = [
        "a nuevo", "a estrenar", "refaccionado",
    ]
    regex = "|".join([fr"\b{palabra}\b" for palabra in keywords])
    sel = (
        df_out["description"].fillna("").str.lower().str.contains(regex, na=False) |
        df_out["title"].fillna("").str.lower().str.contains(regex, na=False)
    )
    df_out["a_nuevo"] = sel
    return df_out

#@title duplex
def duplex(df: pd.DataFrame):
    df_out  = df.copy()
    # A refaccionar
    keywords = [
        "duplex", "dúplex", "triplex", "tríplex"
    ]
    regex = "|".join([fr"\b{palabra}\b" for palabra in keywords])
    sel = (
        df_out["description"].fillna("").str.lower().str.contains(regex, na=False) |
        df_out["title"].fillna("").str.lower().str.contains(regex, na=False)
    )
    df_out["duplex"] = sel
    return df_out

#@title toilette
def toilette(df: pd.DataFrame):
    df_out  = df.copy()
    # A refaccionar
    keywords = [
        "toilette", "toilet", "toillet", "toallet", "toillett", "toalete", "toalet"
    ]
    regex = "|".join([fr"\b{palabra}\b" for palabra in keywords])
    sel = (
        df_out["description"].fillna("").str.lower().str.contains(regex, na=False) |
        df_out["title"].fillna("").str.lower().str.contains(regex, na=False)
    )
    df_out.loc[sel, "bathrooms"] = df_out.loc[sel, "bathrooms"].fillna(0) + 0.5
    return df_out

#@title lujos2
def lujos(df: pd.DataFrame):
    df_out = df.copy()
    keywords = [
        "piscina", "parrilla", "dependencia", "quincho", "terraza", "patio", "pileta", "sauna", "gym|gimnasio"
    ]
    regex = "|".join([fr"\b{palabra}\b" for palabra in keywords])

    desc = df_out["description"].fillna("").str.lower().str.count(regex)
    title = df_out["title"].fillna("").str.lower().str.count(regex)

    df_out["lujos"] = desc + title
    return df_out

#@title puertomadero
def puerto_madero(df: pd.DataFrame):
    df_out = df.copy()
    df_out["es_PuertoMadero"] = (
        (df_out["l3"] == "Puerto Madero") |
        (df_out["title"].str.lower().str.contains("puerto madero", na=False)) |
        (df_out["description"].str.lower().str.contains("puerto madero", na=False))
    )
    return df_out


def aplico_dummies_casa(df: pd.DataFrame):
    df_out = a_refaccionar(df)
    df_out = toilette(df_out)
    df_out = lujos(df_out)
    df_out = duplex(df_out)
    df_out = a_nuevo(df_out)
    df_out = puerto_madero(df_out)
    return df_out


def aplico_dummies_dpto(df: pd.DataFrame):
    df_out = a_refaccionar(df)
    df_out = toilette(df_out)
    df_out = lujos(df_out)
    df_out = duplex(df_out)
    df_out = a_nuevo(df_out)
    df_out = puerto_madero(df_out)
    return df_out

