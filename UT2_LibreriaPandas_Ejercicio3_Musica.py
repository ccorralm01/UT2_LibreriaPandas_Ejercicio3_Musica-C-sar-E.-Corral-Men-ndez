import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('artists.csv')

# Ejercicio 0. Limpieza y Preparación de Datos:
def preparacion_datos(df):
    # Eliminar las filas que contienen campos vacíos en las columnas mbid, artist_mb, artist_lastfm
    df_sin_vacios = df.dropna(subset=['mbid', 'artist_mb', 'artist_lastfm'])

    # Concatenar las dos columnas de nombres de artistas
    df_sin_vacios['combined_artists'] = df_sin_vacios['artist_mb'] + df_sin_vacios['artist_lastfm']

    # Utilizar explode para desglosar la columna de nombres de artistas
    df_exploded_artists = df_sin_vacios.explode('combined_artists')

    # Filtrar filas con nombres de artistas que cumplen con UTF-8
    df_utf8_compatible = df_exploded_artists[df_exploded_artists['combined_artists'].apply(is_utf8_encodable)]

    return df_utf8_compatible

def is_utf8_encodable(s):
    try:
        s.encode('utf-8').decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False



# Ejercicio 1. Análisis Descriptivo Básico:
def analisis_basico(df):

    df_cantidad_artistas = len(df)
    print("Hay un total de", df_cantidad_artistas, "artistas")

    df_media = df['listeners_lastfm'].mean()
    print(f"Lastfm tiene una media de {df_media.round()} oyentes")

    df_mediana = df['listeners_lastfm'].median()
    print(f"Lastfm tiene una mediana de {df_mediana} oyentes")

    df_desviacion = df["scrobbles_lastfm"].std()
    print(f"La desviación estándar del número de oyentes y reproducciones es: {df_desviacion.round()}")

    generar_grafico_cantantes_pais(df, 10, 'bar')


def generar_grafico_cantantes_pais(df, cantidad, tipo_grafo):
    df_paises = df.groupby("country_mb")
    df_paises_count = df_paises["mbid"].count()

    # Crear gráfico

    top_paises = df_paises_count.nlargest(cantidad)

    top_paises.plot(kind=tipo_grafo)

    plt.xlabel('País')
    plt.ylabel('Cantidad de Artistas')
    plt.title('Cantidad de Artistas por País')

    plt.show()


# Funcion que saca las etiquetas y su número de apariciones y lo pasa en una serie
def crear_serie_tags(df_etiquetas):
    etiquetas = {}
    for tags in df_etiquetas:
        if tags is not np.NAN:
            for tag in tags.split(';'):
                tag = tag.strip()
                if tag in etiquetas:
                    etiquetas[tag] += 1
                else:
                    etiquetas[tag] = 1

    return pd.Series(etiquetas)


# Funcion que concatena las etiquetas de mb y fm para su tratamiento posterior y mostrar en gráfico
def analisis_etiquetas(df):

    df_etiquetas_mb = df["tags_mb"]
    df_etiquetas_fm = df["tags_lastfm"]

    etiquetas_concatenadas = pd.concat([df_etiquetas_fm, df_etiquetas_mb])

    serie_etiquetas = crear_serie_tags(etiquetas_concatenadas)

    top_etiquetas = serie_etiquetas.nlargest(10)

    top_etiquetas.plot(kind='pie')

    plt.title('Etiquetas más comunes')
    plt.show()


def combinar_paises_etiquetas(df):
    df['country_lastfm'] = df['country_lastfm'].str.split('; ')

    df_explode_lastfm = df.explode('country_lastfm')
    df_explode_mb = df.explode('country_mb')

    df_explode_lastfm['combined_country'] = df_explode_lastfm['country_lastfm']
    df_explode_mb['combined_country'] = df_explode_mb['country_mb']

    df_explode_lastfm['combined_tags'] = df_explode_lastfm['tags_lastfm']
    df_explode_mb['combined_tags'] = df_explode_mb['tags_mb']

    df_etiquetas_paises = pd.concat([df_explode_lastfm, df_explode_mb])

    return df_etiquetas_paises

def relaciones_etiquetas(df):

    df_etiquetas_paises = combinar_paises_etiquetas(df)

    # Eliminar filas con etiquetas nulas
    df_etiquetas_paises = df_etiquetas_paises.dropna(subset=['combined_tags'])

    # Hacer split a las tags
    df_etiquetas_paises['combined_tags'] = df_etiquetas_paises['combined_tags'].str.split('; ')

    df_tags = df_etiquetas_paises.explode('combined_tags')

    # Contar las etiquetas más comunes por país
    df_tags_count = df_tags.groupby(['combined_country', 'combined_tags']).size().reset_index(name='count')

    # Obtener las etiquetas más comunes por país
    df_top_tags = df_tags_count.groupby('combined_country').apply(lambda x: x.nlargest(5, 'count')).reset_index(
        drop=True)

    # Seleccionar solo los 10 primeros países porque si no es ilegible
    top_countries = df_top_tags['combined_country'].value_counts().nlargest(10).index
    df_top_tags = df_top_tags[df_top_tags['combined_country'].isin(top_countries)]

    # Crear un gráfico de barras horizontal para las etiquetas más comunes por país
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors[:len(top_countries)]
    for country, color in zip(top_countries, colors):
        country_data = df_top_tags[df_top_tags['combined_country'] == country]
        plt.barh(country_data['combined_tags'], country_data['count'], color=color, label=country)

    plt.title('Top 5 Tags más comunes por país (solo los 10 primeros países)')
    plt.xlabel('Count')
    plt.ylabel('Tags')
    plt.legend(title='País', loc='upper right')
    plt.show()


# Funcion que busca ambiguous_artist = true y devuelve los datos de esas filas por cada artista para ver posibles causas de duplicación
def analizar_duplicados_ambiguous(df):
    df = combinar_paises_etiquetas(df)

    artist_duplicates = df[df['ambiguous_artist'] == True]

    # print(artist_duplicates)

    if not artist_duplicates.empty:
        print("Artistas con 'ambiguous_artist' igual a TRUE:")
        print(artist_duplicates[['artist_lastfm', 'combined_country']])

        # Lo que hago en esta función es imprimir los paises y etiquetas de los artistas que son ambiguos para poder
        # entender el motivo por el que se repiten.

        for artist, group in artist_duplicates.groupby('artist_lastfm'):
            print(f"\nCausas posibles para el artista '{artist}':")
            for idx, row in group.iterrows():
                print(f" - País: {row['combined_country']}, Etiquetas: {row['combined_tags']}")

    else:
        print("No se encontraron artistas con 'ambiguous_artist' igual a TRUE.")

# Funcion que comprueba si los datos de Last.fm y MusicBrainz son iguales, si no: mostrar por pantalla.
def comprobar_coincidencias(df):
    # Filtrar las filas donde los datos de Last.fm y MusicBrainz no coinciden
    no_coinciden = df[(df['artist_lastfm'] != df['artist_mb']) |
                      (df['country_lastfm'] != df['country_mb']) |
                      (df['tags_lastfm'] != df['tags_mb'])]

    if not no_coinciden.empty:
        print("Datos que no coinciden entre Last.fm y MusicBrainz:")
        print(no_coinciden[['mbid', 'artist_lastfm', 'artist_mb', 'country_lastfm', 'country_mb', 'tags_lastfm',
                            'tags_mb']])

    else:
        print("Todos los datos coinciden entre Last.fm y MusicBrainz.")


def visualizar_relacion_oyentes_reproducciones(df):

    df = df.dropna(subset=['listeners_lastfm', 'scrobbles_lastfm'])

    # Crear un gráfico de dispersión
    plt.figure(figsize=(10, 6))
    plt.scatter(df['listeners_lastfm'], df['scrobbles_lastfm'], alpha=0.5)
    plt.title('Relación entre el número de oyentes y el número de reproducciones')
    plt.xlabel('Número de oyentes')
    plt.ylabel('Número de reproducciones')
    plt.grid(True)
    plt.show()

def obtener_info_artistas_espana(df):
    df_explode_lastfm = df.explode('country_lastfm')
    df_explode_mb = df.explode('country_mb')

    # Combina las columnas de países en una sola columna
    df_explode_lastfm['combined_country'] = df_explode_lastfm['country_lastfm']
    df_explode_mb['combined_country'] = df_explode_mb['country_mb']

    # Combina las columnas de etiquetas en una sola columna
    df_explode_lastfm['combined_tags'] = df_explode_lastfm['tags_lastfm']
    df_explode_mb['combined_tags'] = df_explode_mb['tags_mb']

    # Combina los DataFrames
    df_combined = pd.concat([df_explode_lastfm, df_explode_mb])

    # Filtrar artistas de España
    artistas_espana = df_combined[df_combined['combined_country'] == 'Spain']

    if not artistas_espana.empty:
        # Encontrar el primer artista de España según sus reproducciones
        primer_artista_espana = artistas_espana.loc[artistas_espana['scrobbles_lastfm'].idxmax()]

        print("Información sobre el primer artista de España según sus reproducciones:")
        print(primer_artista_espana[['artist_lastfm', 'scrobbles_lastfm']])

        # Extraer información sobre las etiquetas más comunes
        etiquetas_espana = ';'.join(artistas_espana['combined_tags'].dropna()).split(';')
        etiquetas_comunes = pd.Series(etiquetas_espana).value_counts().head(10)

        print("\nEtiquetas más comunes de artistas de España:")
        print(etiquetas_comunes)

        # Realizar un gráfico de tarta con la representación de los 10 artistas más escuchados de España
        top_artistas_espana = artistas_espana.nlargest(10, 'scrobbles_lastfm')
        plt.figure(figsize=(10, 6))
        plt.pie(top_artistas_espana['scrobbles_lastfm'], labels=top_artistas_espana['artist_lastfm'], autopct='%1.1f%%', startangle=140)
        plt.title('Top 10 Artistas más escuchados de España')
        plt.show()

    else:
        print("No hay artistas de España en el conjunto de datos.")


# 0. Limpieza y Preparación de Datos:
df_filtrado = preparacion_datos(df)

# 1. Análisis Descriptivo Básico:
analisis_basico(df_filtrado)

# 2. Análisis de Etiquetas:
analisis_etiquetas(df_filtrado)
relaciones_etiquetas(df_filtrado)

# 3. Análisis de Duplicados y consistencia:
analizar_duplicados_ambiguous(df_filtrado)
comprobar_coincidencias(df_filtrado)

# Visualización de Datos 4.:
generar_grafico_cantantes_pais(df_filtrado, 25, 'pie')
generar_grafico_cantantes_pais(df_filtrado, 25, 'bar')

# Visualización de Datos 5.:
visualizar_relacion_oyentes_reproducciones(df_filtrado)

# Visualización de Datos 6.:
obtener_info_artistas_espana(df_filtrado)
