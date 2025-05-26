import requests
import zipfile
import geopandas as gpd
import io
import numpy as np
import pandas as pd

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


def invierto_lat_lon(df):
    df_out = df.copy()
    df_out[["lon", "lat"]] = df_out[["lat", "lon"]].values
    df_out["lat"] = df_out["lat"].replace(1, np.nan)
    return df_out


# @title me_fijo_si_barrio_esta_bien
def me_fijo_si_barrio_esta_bien(df):
    df_out = df.copy()
    # Paso 1: Descargar y descomprimir el shapefile
    url = "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall("barrios_caba")

    barrios_oficial = gpd.read_file("barrios_caba/barrios.shp")

    gdf = gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )

    barrios_oficial = barrios_oficial.to_crs("EPSG:4326")
    barrios_oficial["nombre"] = barrios_oficial["nombre"].str.lower()
    gdf["l3"] = gdf["l3"].str.lower()
    gdf["l3"] = gdf["l3"].str.replace("agronomía", "agronomia")
    gdf["l3"] = gdf["l3"].str.replace("constitución", "constitucion")
    gdf["l3"] = gdf["l3"].str.replace("villa general mitre", "villa gral. mitre")
    gdf["l3"] = gdf["l3"].str.replace("villa pueyrredón", "villa pueyrredon")

    # Paso 4: Hacer un spatial join
    gdf_joined = gpd.sjoin(
        gdf, barrios_oficial[["nombre", "geometry"]], how="left", predicate="within"
    )

    df_out["barrio_oficial"] = gdf_joined["nombre"]
    df_out["l3"] = df_out["l3"].str.lower()

    df_out["is_barrio_ok"] = df_out["l3"] == df_out["barrio_oficial"]
    return df_out


def releno_l3_con_barrio_oficial(df):
    df_out = df.copy()
    df_out["l3"] = df_out["l3"].str.lower()
    df_out["barrio_oficial"] = df_out["barrio_oficial"].str.lower()
    df_out["l3"] = df_out["l3"].fillna(df_out["barrio_oficial"])
    return df_out


def relleno_latlon_con_media_barrio(df, by="barrio_oficial"):
    df_out = df.copy()
    # Calculo la medi a delos barrios
    media_latlon_barrios = df_out.groupby(by).agg(
        mean_lat=("lat", "mean"),
        mean_lon=("lon", "mean"),
    )
    df_out = df_out.merge(
        media_latlon_barrios, left_on=by, right_index=True, how="left"
    )
    # Flag
    df_out["corrijo_latlon"] = df_out["lat"].isna() | (df_out["lon"].isna())
    df_out["lat"] = df_out["lat"].fillna(df_out["mean_lat"])
    df_out["lon"] = df_out["lon"].fillna(df_out["mean_lon"])
    # Si barrio_oficial es nan, pongo l3
    df_out["barrio_oficial"] = df_out["barrio_oficial"].fillna(df_out["l3"])
    return df_out


def barrios_con_OSM(df, by="barrio_oficial"): #, barrio="belgrano"):
    geolocator = Nominatim(user_agent="sdfghj")
    reverse = RateLimiter(
        geolocator.reverse, min_delay_seconds=1, max_retries=3, error_wait_seconds=2.0
    )

    # Cache para evitar duplicar consultas
    cache = {}

    def get_osm_info(lat, lon):
        if np.isnan(lat) or np.isnan(lon):
            return (None, None)
        key = (round(lat, 5), round(lon, 5))  # redondeo para evitar microvariaciones
        if key in cache:
            return cache[key]

        try:
            location = reverse((lat, lon), exactly_one=True, language="es")
            address = location.raw.get("address", {}) if location else {}
            suburb = address.get("suburb")
            neighbourhood = address.get("neighbourhood")
            result = (suburb, neighbourhood)
            print(f"{lat=}, {lon=} -> {result=}")
        except Exception as e:
            print(f"Error: {e}")
            result = (None, None)

        cache[key] = result
        return result

    def agregar_barrios_osm(df):
        """Agrega columnas 'l3_OSM' y 'l4_OSM' usando geopy y cache."""
        df[["l3_OSM", "l4_OSM"]] = df.apply(
            lambda row: pd.Series(get_osm_info(row["lat"], row["lon"])), axis=1
        )
        return df

    df_out = df.copy()
    aver = agregar_barrios_osm(df_out)
    aver = aver[["lat", "lon", "l3_OSM", "l4_OSM"]]
    aver.to_csv("barrios_osm.csv", index=False)
    return aver


