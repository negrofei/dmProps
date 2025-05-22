import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from datetime import datetime

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

import time


def ver_correlaciones(
    df,
    metodo="pearson",
    figsize=(10, 8),
    vmin=-1,
    vmax=1,
    cmap="coolwarm",
    mask_upper=True,
):
    """
    Muestra un mapa de calor con las correlaciones entre los features numéricos del DataFrame.

    Parámetros:
    - df: DataFrame de entrada.
    - metodo: Método de correlación. Puede ser "pearson", "spearman" o "kendall".
    - figsize: Tamaño de la figura.
    - vmin, vmax: Rango de valores del color.
    - cmap: Colormap a usar.
    - mask_upper: Si True, oculta la mitad superior de la matriz.
    """

    corr = df.corr(method=metodo)

    # Opcional: ocultar la parte superior del mapa de calor
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
    )
    plt.title(f"Matriz de correlación ({metodo})")
    plt.tight_layout()
    plt.show()


# @title `comparo_test_train`
def comparo_test_train(df_train, df_test, cols="all", show=False):

    columnas = [
        ("l3", None, None),
        ("barrio", None, None),
        ("rooms", 0.9995, 1),
        ("bedrooms", 0.9995, 1),
        ("bathrooms", 0.9995, 1),
        ("surface_total", 0.99, 20),
        ("surface_covered", 0.99, 20),
        ("surface_mean", 0.99, 20),
        ("surface_sum", 0.99, 20),
        ("m2_x_amb", 0.995, 10),
        ("sup_x_room", 0.99, 5),
        ("total-cov", 0.99, 20),
        ("room-bed", 0.99, 1),
    ]
    if cols == "all":
        columnas_filtradas = [
            c
            for c in columnas
            if (c[0] in df_train.columns) and (c[0] in df_test.columns)
        ]
    else:
        columnas_filtradas = [
            c
            for c in columnas
            if (c[0] in cols)
            and (c[0] in df_train.columns)
            and (c[0] in df_test.columns)
        ]

    for col, q, delta in columnas_filtradas:

        plt.figure(figsize=(10, 6))
        if q == None:
            df_test[col].hist(
                bins=len(df_test[col].unique()),
                alpha=0.6,
                density=True,
                label="df_test",
            )
            df_train[col].hist(
                bins=len(df_train[col].unique()),
                alpha=0.6,
                density=True,
                label="df_train",
            )
        else:
            df_test[col].hist(
                bins=np.arange(0, df_test[col].quantile(q), delta),
                alpha=0.6,
                density=True,
                label="df_test",
            )

            df_train[col].hist(
                bins=np.arange(0, df_train[col].quantile(q), delta),
                alpha=0.6,
                density=True,
                label="df_train",
            )
        plt.title(f"Distribución de {col}")
        plt.xlabel(col)
        plt.ylabel("Densidad")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"hist_{col}.png")  # Guarda la figura como imagen PNG
        if show:
            plt.show()


# @title figura_barrios_geo
def figura_barrios_geo(df, by="l3"):
    fig = px.scatter_mapbox(
        df.sort_values(by=by),
        lat="lat",
        lon="lon",
        color=by,
        mapbox_style="carto-positron",
    )

    fig.show()


# @title grafico_4d_lofScore_y_filtroThr
def grafico_4d_lofScore_y_filtroThr(
    df, x_col, y_col, size_col, color_col, lof_col, thr, symmetric_axes=False
):
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(8, 6), sharex=True)
    if size_col is not None:
        scaler = MinMaxScaler()
        size = scaler.fit_transform(df[[size_col]])
    else:
        size = 0.1
    axes = axes.ravel()
    ax = axes[0]
    asd = ax.scatter(
        df[x_col], df[y_col], s=size * 100, c=df[color_col], cmap="viridis", alpha=0.8
    )
    plt.colorbar(mappable=asd, label=color_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    filtrado = df[df[lof_col] > thr]
    if size_col is not None:
        size = scaler.fit_transform(filtrado[[size_col]])
    else:
        size = 0.1

    ax = axes[1]
    asd2 = ax.scatter(
        x=filtrado[x_col],
        y=filtrado[y_col],
        s=size * 100,
        c=filtrado[color_col],
        cmap="viridis",
        alpha=0.8,
    )
    plt.colorbar(mappable=asd2, label=color_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if symmetric_axes:
        for ax in axes:
            ax.set_aspect("equal")
        # # calcular límites comunes
        # all_x = pd.concat([df[x_col], filtrado[x_col]])
        # all_y = pd.concat([df[y_col], filtrado[y_col]])
        # xlim = [all_x.min(), all_x.max()]
        # ylim = [all_y.min(), all_y.max()]
        # for ax in axes:
        #     ax.set_xlim(xlim)
        #     ax.set_ylim(ylim)

    return filtrado


# @title grafico_4d_lado_a_lado
def grafico_4d_lado_a_lado(
    df_train, df_test, x_col, y_col, size_col=None, color_col=None
):
    fig, axes = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(10, 5))
    axes = axes.ravel()

    # Escalar tamaño si corresponde
    def escalar_tamanio(df):
        if size_col is not None:
            scaler = MinMaxScaler()
            return scaler.fit_transform(df[[size_col]]) * 100
        return 30

    size_train = escalar_tamanio(df_train)
    size_test = escalar_tamanio(df_test)

    # Unificar escala de colores si corresponde
    norm = None
    cmap = None
    if color_col:
        vmin = min(df_train[color_col].min(), df_test[color_col].min())
        vmax = max(df_train[color_col].max(), df_test[color_col].max())
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = "viridis"

    # Dibujar ambos
    sc = None
    for i, (df, size, titulo) in enumerate(
        zip([df_train, df_test], [size_train, size_test], ["Train", "Test"])
    ):
        ax = axes[i]
        sc = ax.scatter(
            df[x_col],
            df[y_col],
            s=size,
            c=df[color_col] if color_col else "blue",
            cmap=cmap,
            norm=norm,
            alpha=0.8,
        )
        ax.set_title(titulo)
        ax.set_xlabel(x_col)
        if i == 0:
            ax.set_ylabel(y_col)

    # Colorbar común
    if color_col:
        cbar = fig.colorbar(
            sc, ax=axes, orientation="vertical", fraction=0.03, pad=0.04
        )
        cbar.set_label(color_col)

    # plt.tight_layout()
    plt.show()


# @title `nans_x_fila` y `nans_x_columns`
def nans_x_fila(df, tipo="train"):
    print(df.shape)
    print(df.isna().sum(axis=1).sort_values())


def nans_x_columna(df, tipo="train", density=True):
    print(df.shape)
    nans = df.isna().sum()
    if density:
        nans_percent = (nans / len(df)) * 100
        nans_df = pd.DataFrame({
            'n_nans': nans,
            'pct_nans': nans_percent.round(2)
        })
        print(nans_df)
    else:
        print(nans)



# @title calculo_LOF
def calculo_LOF(
    df,
    columnas=["rooms", "bedrooms", "bathrooms"],
    nombre="lof_score",
    tipo="train",
    prop_vecinos=0.05,
):
    copia = df.copy()
    subset = copia[columnas].dropna().drop_duplicates().copy()

    # Ajuste LOF
    vecinos = min(max(5, int(len(subset) * prop_vecinos)), 1000)
    print(f"Tomo {vecinos} vecinos")
    subset[nombre] = (
        -LocalOutlierFactor(n_neighbors=vecinos, contamination="auto")
        .fit(subset)
        .negative_outlier_factor_
    )
    for col in subset.columns:
        subset[f"{col}_q"] = subset[col].rank(pct=True)

    return subset.sort_values(by=nombre, ascending=False)


def elapsed_time(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        elapsed = end_time - start_time
        elapsed_str = str(elapsed).split(".")[0]  # Format as H:MM:SS
        print(f"Tiempo de ejecución: {elapsed_str}")
        return result

    return wrapper
