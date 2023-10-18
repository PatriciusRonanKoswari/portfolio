from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import sys
import re
import itertools
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
from collections import Counter
from skimage import io

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

song_data = pd.read_csv('data-2.csv')

song_data['artists_song'] = song_data['artists_name'] +' ' + '-' + song_data['track_name']


genre_data = pd.read_csv('genres-2.csv')


genre_data['genres'] = genre_data['genres'].apply(lambda x: [i for i in re.findall(r"'([^']*)'", x)])

song_data.drop_duplicates('artists_song', inplace = True)
artists_explode = song_data[['artists_name','track_id']].explode('artists_name')


artists_explode.head()

artists_explode = artists_explode.merge(genre_data, how = 'left', left_on = 'artists_name',right_on = 'artists')
artists_explode= artists_explode.dropna(subset=['genres'])


artists_explode.head(7)


artists_genres_combine = artists_explode.groupby('track_id')['genres'].apply(list).reset_index()
artists_genres_combine.head()


artists_genres_combine['genres'] = artists_genres_combine['genres'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))


artists_genres_combine.head()


song_data = song_data.merge(artists_genres_combine[['track_id','genres']],on='track_id', how='left')


song_data.head()


float_column = song_data.dtypes[song_data.dtypes == 'float64'].index.values


song_data['genres'] = song_data['genres'].apply(lambda y: y if isinstance(y,list) else [])


onehotencoding_cols = 'popularity'


song_data['popularity_rating'] = song_data['popularity'].apply(lambda x: int(x/5))


def OneHotEncoding(data, column, new_name):
    TF = pd.get_dummies(data[column])
    feature_names = TF.columns
    TF.columns = [new_name + "||" + str(i) for i in feature_names]
    TF.reset_index(inplace=True, drop=True)
    return TF

def create_feature_set(data, float_cols):

    TfIdf = TfidfVectorizer()
    TfIdf_matrix =  TfIdf.fit_transform(data['genres'].apply(lambda x: " ".join(x)))
    genre_data = pd.DataFrame(TfIdf_matrix.toarray())
    genre_data.columns = ['genre' + "|" + i for i in TfIdf.get_feature_names_out()]
    genre_data.reset_index(drop = True, inplace=True)

    floats = data[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2
    OneHotEncoding_year = OneHotEncoding(data, 'year','year') * 0.5
    OneHotEncoding_pop = OneHotEncoding(data,'popularity_rating','pop') * 0.2

    final = pd.concat([genre_data, floats_scaled, OneHotEncoding_pop, OneHotEncoding_year], axis = 1)
     
    final['track_id']=data['track_id'].values
    
    return final

complete_feature_set = create_feature_set(song_data, float_cols=float_column)

complete_feature_set.head(55)

client_id = '71b8a60e3de244d6a14f3ddbb3532b4e'
client_secret= '78ad8c957e954325b07e3e37a8c139d5'
scope = 'user-library-read'

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)
token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret, redirect_uri='http://localhost:5555/callback')
sp = spotipy.Spotify(auth=token)

name_and_id = {}
photo_list = {}

for i in sp.current_user_playlists()['items']:
    name_and_id[i['name']] = i['uri'].split(':')[2]
    photo_list[i['uri'].split(':')[2]] = i['images'][0]['url']



def create_necessary_output(playlist_name,id_map,df):

    playlist = pd.DataFrame()
    playlist_name = playlist_name

    for x, i in enumerate (sp.playlist(id_map[playlist_name])['tracks']['items']):

        playlist.loc[x, 'artist'] = i['track']['artists'][0]['name']
        playlist.loc[x, 'name'] = i['track']['name']
        playlist.loc[x, 'id'] = i['track']['id'] 
        playlist.loc[x, 'url'] = i['track']['album']['images'][1]['url']
        playlist.loc[x, 'date_added'] = i['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])  
    playlist = playlist[playlist['id'].isin(df['track_id'].values)].sort_values('date_added',ascending = False)

    return playlist


playlist_90an = create_necessary_output('2010-2020 Top Hits (2010’s) ', name_and_id, song_data)



def visualize_playlist(df):

    temp =  df['url'].values
    plt.figure(figsize=(15, int(math.ceil(0.7 * len(temp)))))
    column = 5

    for i, url in enumerate(temp): 
        plt.subplot(math.ceil(len(temp) / column), column, i + 1)

        image = io.imread(url)
        plt.imshow(image)
        plt.xticks(color='w', fontsize=0.1)
        plt.yticks(color='w', fontsize=0.1)
        plt.xlabel(df['name'].values[i], fontsize=10)
        plt.tight_layout(h_pad=1, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()
    


def generate_playlist_feature(complete_feature_set, playlist_df):
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['track_id'].isin(playlist_df['id'].values)]

    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['track_id'].isin(playlist_df['id'].values)]
    complete_feature_set_playlist_final = complete_feature_set_playlist.drop(columns = "track_id")

    return complete_feature_set_playlist_final.sum(axis = 0), complete_feature_set_nonplaylist


complete_feature_set_playlist_vector, complete_feature_set_nonplaylist = generate_playlist_feature(complete_feature_set, playlist_90an)


def finalize_recommendation(df, features, nonplaylist_features):
    non_playlist_df = df[df['track_id'].isin(nonplaylist_features['track_id'].values)]
    non_playlist_df['cosine_sim'] = cosine_similarity(nonplaylist_features.drop('track_id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_5 = non_playlist_df.sort_values('cosine_sim',ascending = False).head(5)
    non_playlist_df_top_5['url'] = non_playlist_df_top_5['track_id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
    
    return non_playlist_df_top_5



playlist_top5 = finalize_recommendation(song_data, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist)




@app.route('/')
def index():
    return render_template('spotify_form.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Mendapatkan daftar playlist dari akun pengguna
    playlists = sp.current_user_playlists()

    # Menyimpan daftar playlist dalam variabel playlist_names
    playlist_names = [f"{playlist['name']}|{playlist['images'][0]['url']}" for playlist in playlists['items']]

    return render_template('success.html', playlist_names=playlist_names)


@app.route('/playlist/<playlist_name>')

def show_playlist(playlist_name):
    # Di sini, Anda akan mendapatkan nama playlist yang dipilih dari URL
    # Kemudian Anda dapat menggunakan nama playlist tersebut untuk menampilkan rekomendasi
    playlist_90an = create_necessary_output(playlist_name, name_and_id, song_data)
    complete_feature_set_playlist_vector_hiphop, complete_feature_set_nonplaylist_hiphop = generate_playlist_feature(complete_feature_set, playlist_90an)
    playlist_top5 = finalize_recommendation(song_data, complete_feature_set_playlist_vector_hiphop, complete_feature_set_nonplaylist_hiphop)

    # Menampilkan hasil rekomendasi di halaman playlist.html
    return render_template('playlist.html', playlist_top5=playlist_top5)

@app.route('/success')
def success():
    return render_template('playlist.html', playlist_top5=playlist_top5)

if __name__ == '__main__':
    app.run(debug=True)



