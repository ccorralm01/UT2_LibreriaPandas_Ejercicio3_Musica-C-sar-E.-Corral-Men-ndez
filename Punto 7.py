import numpy as np
import pandas as pd
from flask import Flask, render_template, request

df = pd.read_csv('artists.csv')
app = Flask(__name__)

"""
He hecho un buscador de artistas a traves de sus etiquetas. He aplicado los mismos filtros que en el punto 0 del
ejercicio, he juntado las etiquetas en una sola columna para tener todas en cuenta. Como la informacion adicional
de oyentes y reproducciones es de lastfm, he tenido más en cuenta el pais de lastfm pero he añadido que en caso de ser
nan se utilice el de mb. Solo se muestran los primeros 50 artistas para optimizar en rendimiento.
"""

def buscar_artista_por_etiquetas_filtar(df, etiquetas):
    # Eliminar las filas que contienen campos vacíos en las columnas mbid, artist_mb, artist_lastfm
    df_sin_vacios = df.dropna(subset=['mbid', 'artist_mb', 'artist_lastfm'])

    # Concatenar las dos columnas de nombres de artistas
    df_sin_vacios['combined_artists'] = df_sin_vacios['artist_mb'] + df_sin_vacios['artist_lastfm']

    # Utilizar explode para desglosar la columna de nombres de artistas
    df_exploded_artists = df_sin_vacios.explode('combined_artists')

    # Filtrar filas con nombres de artistas que cumplen con UTF-8
    df_utf8_compatible = df_exploded_artists[df_exploded_artists['combined_artists'].apply(is_utf8_encodable)]

    # Concatenar las etiquetas de lastfm y musicbrainz
    df['combined_tags'] = df['tags_lastfm'] + ';' + df['tags_mb']

    # Filtrar el DataFrame para obtener las filas que contienen las etiquetas deseadas
    etiquetas_lista = etiquetas.split(',')
    df_filtrado = df[df['combined_tags'].str.contains('|'.join(etiquetas_lista), case=False, na=False)]

    # Comprobar y asignar country_mb si country_lastfm es NaN
    df_filtrado['country_lastfm'].fillna(df_filtrado['country_mb'], inplace=True)

    # Limitar los resultados a los primeros 50
    df_filtrado = df_filtrado.head(50)

    return df_filtrado, etiquetas_lista


def is_utf8_encodable(s):
    try:
        s.encode('utf-8').decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        etiquetas_usuario = request.form['etiqueta']
        df_filtrado, etiquetas_lista = buscar_artista_por_etiquetas_filtar(df, etiquetas_usuario)
        return render_template('index.html', df_filtrado=df_filtrado.to_dict(orient='records'),
                               etiquetas_lista=etiquetas_lista)
    else:
        return render_template('index.html', df_filtrado=None, etiquetas_lista=None)

if __name__ == '__main__':
    app.run(debug=True)