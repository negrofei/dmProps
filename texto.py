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