from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import json
import re 
import sys
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import warnings
warnings.filterwarnings("ignore")

from skimage import io
import matplotlib.pyplot as plt
import math

app = Flask(__name__)

spotify_df = pd.read_csv('data.csv')


data_w_genre = pd.read_csv('data_w_genres.csv')
data_w_genre.dtypes
data_w_genre['genres'].values[0]
data_w_genre['genres'].values[0][0]


data_w_genre['genres_upd'] = data_w_genre['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])
data_w_genre['genres_upd'].values[0][0]
spotify_df['artists_upd_v1'] = spotify_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))
spotify_df['artists'].values[0]
spotify_df['artists_upd_v1'].values[0][0]

spotify_df[spotify_df['artists_upd_v1'].apply(lambda x: not x)].head(5)
spotify_df['artists_upd_v2'] = spotify_df['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
spotify_df['artists_upd'] = np.where(spotify_df['artists_upd_v1'].apply(lambda x: not x), spotify_df['artists_upd_v2'], spotify_df['artists_upd_v1'] )
spotify_df['artists_song'] = spotify_df.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)
spotify_df.sort_values(['artists_song','release_date'], ascending = False, inplace = True)


spotify_df[spotify_df['name']=='Adore You']
spotify_df.drop_duplicates('artists_song',inplace = True)
spotify_df[spotify_df['name']=='Adore You']
artists_exploded = spotify_df[['artists_upd','id']].explode('artists_upd')
artists_exploded_enriched = artists_exploded.merge(data_w_genre, how = 'left', left_on = 'artists_upd',right_on = 'artists')
artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]


artists_exploded_enriched_nonnull[artists_exploded_enriched_nonnull['id'] =='6KuQTIu1KoTTkLXKrwlLPV']
artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()
artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))
artists_genres_consolidated.head()


spotify_df = spotify_df.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')
spotify_df.tail()
spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])
float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values


ohe_cols = 'popularity'
spotify_df['popularity'].describe()
spotify_df['popularity_red'] = spotify_df['popularity'].apply(lambda x: int(x/5))
spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])
spotify_df.head()


def one_hot_encoding_prep(data, column, new_name): 
    tf_data = pd.get_dummies(data[column])
    feature_names = tf_data.columns
    tf_data.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_data.reset_index(drop = True, inplace = True)    
    return tf_data


def create_feature_set(data, float_cols):

    TfIdf = TfidfVectorizer()
    TfIdf_matrix =  TfIdf.fit_transform(data['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
    genre_data = pd.DataFrame(TfIdf_matrix.toarray())
    genre_data.columns = ['genre' + "|" + i for i in TfIdf.get_feature_names_out()]
    genre_data.reset_index(drop = True, inplace=True)

    year_one_hot_encoding = one_hot_encoding_prep(data, 'year','year') * 0.5
    popularity_one_hot_encoding = one_hot_encoding_prep(data, 'popularity_red','pop') * 0.15

    floats = data[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    final = pd.concat([genre_data, floats_scaled, popularity_one_hot_encoding, year_one_hot_encoding], axis = 1)
     
    final['id']=data['id'].values
    
    return final

complete_feature_set = create_feature_set(spotify_df, float_cols=float_cols)
complete_feature_set.head()

client_id = '71b8a60e3de244d6a14f3ddbb3532b4e'
client_secret= '78ad8c957e954325b07e3e37a8c139d5'
scope = 'user-library-read'

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)
token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret, redirect_uri='http://localhost:5555/callback')
sp = spotipy.Spotify(auth=token)
id_name = {}
list_photo = {}
for i in sp.current_user_playlists()['items']:

    id_name[i['name']] = i['uri'].split(':')[2]
    list_photo[i['uri'].split(':')[2]] = i['images'][0]['url']


def create_necessary_outputs(playlist_name, id_dic, df):
    #generate playlist dataframe
    playlist = pd.DataFrame()
    playlist_name = playlist_name
    for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
        playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
        playlist.loc[ix, 'name'] = i['track']['name']
        playlist.loc[ix, 'id'] = i['track']['id']
        playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
        playlist.loc[ix, 'date_added'] = i['added_at']
    playlist['date_added'] = pd.to_datetime(playlist['date_added'])   
    playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added', ascending=False)   
    return playlist

playlist_hiphop = create_necessary_outputs('$$', id_name,spotify_df)



def visualize_songs(df):
    temp = df['url'].values
    plt.figure(figsize=(15, int(math.ceil(0.625 * len(temp)))))
    columns = 5
    for i, url in enumerate(temp):
        plt.subplot(math.ceil(len(temp) / columns), columns, i + 1)
        image = io.imread(url)
        plt.imshow(image)
        plt.xticks(color='w', fontsize=0.1)
        plt.yticks(color='w', fontsize=0.1)
        plt.xlabel(df['name'].values[i], fontsize=12)
        plt.tight_layout(h_pad=0.4, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)
    plt.show()

def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    playlist_feature_set = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]
    playlist_feature_set = playlist_feature_set.merge(playlist_df[['id','date_added']], on='id', how='inner')
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]
    playlist_feature_set = playlist_feature_set.sort_values('date_added', ascending=False)
    most_recent_date = playlist_feature_set.iloc[0,-1]
    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix, 'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x)) 
    playlist_feature_set_weighted = playlist_feature_set.copy()
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
    
    return playlist_feature_set_weighted_final.sum(axis=0), complete_feature_set_nonplaylist
complete_feature_set_playlist_vector_hiphop, complete_feature_set_nonplaylist_hiphop = generate_playlist_feature(complete_feature_set, playlist_hiphop, 1.09)
complete_feature_set_playlist_vector_hiphop.shape


def generate_playlist_recos(df, features, nonplaylist_features):
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_40 = non_playlist_df.sort_values('sim',ascending = False).head(40)
    non_playlist_df_top_40['url'] = non_playlist_df_top_40['id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
    
    return non_playlist_df_top_40

hiphop_top40 = generate_playlist_recos(spotify_df, complete_feature_set_playlist_vector_hiphop, complete_feature_set_nonplaylist_hiphop)


@app.route('/')
def index():
    return render_template('spotify_form.html')

@app.route('/submit', methods=['POST'])
def submit():
    client_id = request.form['client_id']
    client_secret = request.form['client_secret']
    client_uri = request.form['client_uri']

    # Mendapatkan token akses Spotify
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    token = util.prompt_for_user_token(scope, client_id=client_id, client_secret=client_secret, redirect_uri=client_uri)
    sp = spotipy.Spotify(auth=token)

    # Mendapatkan daftar playlist dari akun pengguna
    playlists = sp.current_user_playlists()

    # Menyimpan daftar playlist dalam variabel playlist_names
    playlist_names = [playlist['name'] for playlist in playlists['items']]

    return render_template('success.html', playlist_names=playlist_names)

@app.route('/playlist/<playlist_name>')
def show_playlist(playlist_name):
    # Di sini, Anda akan mendapatkan nama playlist yang dipilih dari URL
    # Kemudian Anda dapat menggunakan nama playlist tersebut untuk menampilkan rekomendasi
    playlist_hiphop = create_necessary_outputs(playlist_name, id_name, spotify_df)
    complete_feature_set_playlist_vector_hiphop, complete_feature_set_nonplaylist_hiphop = generate_playlist_feature(complete_feature_set, playlist_hiphop, 1.09)
    hiphop_top40 = generate_playlist_recos(spotify_df, complete_feature_set_playlist_vector_hiphop, complete_feature_set_nonplaylist_hiphop)

    # Menampilkan hasil rekomendasi di halaman playlist.html
    return render_template('playlist.html', hiphop_top40=hiphop_top40)

@app.route('/success')
def success():
    return render_template('playlist.html', hiphop_top40=hiphop_top40)

if __name__ == '__main__':
    app.run(debug=True)
