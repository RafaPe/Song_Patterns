import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stat
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap

st.title("Song features Analysis of trending songs ♫")
st.sidebar.title("Song features Analysis of trending songs ♫")

st.markdown("This application is a Streamlit dashboard to analyze the song features and data of our dataset")
st.sidebar.markdown("This application is a Streamlit dashboard to analyze the song features and data of our dataset")

DATA_URL=("data/song-features-v2.csv")
DATA_URL_RAFA=("data/songsrafa.csv")
DATA_URL_FER=("data/songsfer.csv")

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL, encoding = "ISO-8859-1")
    return data
def load_dataraf():
    data = pd.read_csv(DATA_URL_RAFA)
    return data
def load_datafer():
    data = pd.read_csv(DATA_URL_FER)
    return data

data=load_data()
data_rafa=load_dataraf()
data_fer=load_datafer()

#Bar plots, of general stats

st.sidebar.markdown("### General stats")

select=st.sidebar.selectbox('Select atribute',['Gender of the artist(s)', 'Type of agrupation','Language','Country'], key='1')

gender_count=data['gender(of the atrist(s))'].value_counts()
gender_count=pd.DataFrame({'Gender':gender_count.index,'Songs':gender_count.values})

type_count=data['type(solo, colab, grpup/band)'].value_counts()
type_count = pd.DataFrame({'Type':type_count.index,'Songs':type_count.values})

lang_count=data['song_language'].value_counts()
lang_count = pd.DataFrame({'Language':lang_count.index,'Songs':lang_count.values})

country_count=data['country'].value_counts()
country_count = pd.DataFrame({'Country':country_count.index,'Songs':country_count.values})

#Rafa's stats
gender_count_r=data_rafa['gender(of the atrist(s))'].value_counts()
gender_count_r=pd.DataFrame({'Gender':gender_count_r.index,'Songs':gender_count_r.values})

type_count_r=data_rafa['type(solo, colab, grpup/band)'].value_counts()
type_count_r = pd.DataFrame({'Type':type_count_r.index,'Songs':type_count_r.values})

lang_count_r=data_rafa['song_language'].value_counts()
lang_count_r = pd.DataFrame({'Language':lang_count_r.index,'Songs':lang_count_r.values})

country_count_r=data_rafa['country'].value_counts()
country_count_r = pd.DataFrame({'Country':country_count_r.index,'Songs':country_count_r.values})

#Fer's stats
gender_count_f=data_fer['gender(of the atrist(s))'].value_counts()
gender_count_f=pd.DataFrame({'Gender':gender_count_f.index,'Songs':gender_count_f.values})

type_count_f=data_fer['type(solo, colab, grpup/band)'].value_counts()
type_count_f = pd.DataFrame({'Type':type_count_f.index,'Songs':type_count_f.values})

lang_count_f=data_fer['song_language'].value_counts()
lang_count_f = pd.DataFrame({'Language':lang_count_f.index,'Songs':lang_count_f.values})

country_count_f=data_fer['country'].value_counts()
country_count_f = pd.DataFrame({'Country':country_count_f.index,'Songs':country_count_f.values})


if not st.sidebar.checkbox("Hide",True):
    if select == "Gender of the artist(s)":
        st.markdown("### General stats, combined sets, Gender of the artist(s)")
        fig = px.bar(gender_count,x='Gender',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig)
        st.markdown("### General stats, Rafa's set, Gender of the artist(s)")
        fig2 = px.bar(gender_count_r,x='Gender',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig2)
        st.markdown("### General stats, Fer's set, Gender of the artist(s)")
        fig3 = px.bar(gender_count_f,x='Gender',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig3)
        
    if select == "Type of agrupation":
        st.markdown("### General stats, combined sets, Type of agrupation")
        fig=px.bar(type_count,x='Type',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig)
        st.markdown("### General stats, Rafa's set, Type of agrupation")
        fig2 = px.bar(type_count_r,x='Type',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig2)
        st.markdown("### General stats, Fer's set, Type of agrupation")
        fig3 = px.bar(type_count_f,x='Type',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig3)
        
    if select == "Language":
        st.markdown("### General stats, combined sets, Language of the songs")
        fig=px.bar(lang_count,x='Language',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig)
        st.markdown("### General stats, Rafa's set, Language of the songs")
        fig2 = px.bar(lang_count_r,x='Language',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig2)
        st.markdown("### General stats, Fer's set, Language of the songs")
        fig3 = px.bar(lang_count_f,x='Language',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig3)
        
    if select == "Country":
        st.markdown("### General stats, combined sets, Country")
        fig=px.bar(country_count,x='Country',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig)
        st.markdown("### General stats, Rafa's set, Country")
        fig2 = px.bar(country_count_r,x='Country',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig2)
        st.markdown("### General stats, Fer's set, Country")
        fig3 = px.bar(country_count_f,x='Country',y='Songs',color='Songs',height=500)
        st.plotly_chart(fig3)
        
st.sidebar.subheader("Map")
    
        
st.sidebar.subheader("Breakdown Song by its technical features")
choice = st.sidebar.multiselect('Pick TWO features',('danceability','energy','key','loudness','valence'),key='0')


if len(choice)==2:
    st.markdown("### Dispersion of the songs given two of its features")
    ch1=data[choice[0]]
    ch2=data[choice[1]]
    fig_choice=px.scatter(data,x=choice[0],y=choice[1],height=600, width=800)
    st.write(fig_choice)

st.sidebar.subheader("Country of the artist(s)")
choice = st.sidebar.multiselect('Pick contry',('England', 'United States', 'Canada', 'Spain', 'France', 'Chile',
       'Puerto Rico', 'Argentina', 'Mexico', 'Colombia', 'Peru', 'Norway',
       'Germany', 'Japan', 'Australia', 'Sweden', 'Nigeria', 'Malaysia',
       'Korea', 'Venezuela', '-', 'Panama', 'Ireland', 'Zimbawe',
       'Switzerland', 'New Zeland', 'Sudan', 'Dominican Republic',
       'Belarus', 'Iran'),key='2')

if len(choice)>0:
    choice_data=data[data.country.isin(choice)]
    fig_choice=px.histogram(choice_data,x='country',y='gender(of the atrist(s))',histfunc='count',color='gender(of the atrist(s))',
    facet_col='gender(of the atrist(s))',labels={'gender(of the atrist(s))':'Gender'}, height=600, width=800)
    st.plotly_chart(fig_choice)


st.sidebar.subheader("Cluster Analysis")
cluster_hide = st.sidebar.checkbox('Hide', True, key='Ck123')
if not cluster_hide:
    songs_from = st.sidebar.radio('Select', ('Fer Songs','Rafa Songs','Both Songs'))


    if songs_from == 'Fer Songs':
        col1, col2 = st.beta_columns(2)
        cluster = st.sidebar.radio('Select', ('Cluster1','Cluster2','Cluster3','Cluster4','Cluster5','Cluster6','Cluster7','Cluster8'))
        songs = data[(data.label == 'F') | (data.label == 'RF') | (data.label == 'FR')]
        # print(songs)
        cluster_lab = [4, 6, 1, 4, 4, 6, 4, 3, 7, 6, 5, 1, 5, 4, 6, 4, 1, 3, 6, 1, 1, 6,
            4, 4, 3, 1, 1, 4, 1, 4, 1, 3, 1, 1, 3, 4, 6, 3, 1, 4, 5, 5, 5, 4,
            4, 0, 6, 4, 6, 6, 6, 1, 5, 1, 7, 4, 4, 1, 6, 3, 1, 6, 4, 5, 6, 4,
            5, 6, 6, 4, 3, 4, 6, 3, 6, 1, 5, 6, 4, 4, 6, 4, 5, 6, 6, 1, 4, 6,
            3, 7, 6, 5, 5, 7, 1, 3, 1, 6, 6, 6, 4, 3, 1, 7, 6, 3, 5, 4, 1, 7,
            4, 4, 4, 4, 2, 4, 4, 4, 4, 7, 3, 4, 6, 1, 3, 1, 3, 7, 5, 4, 3, 3,
            3, 3, 3, 5, 1, 0, 6, 6, 6, 0, 6, 4, 1, 4, 5, 1, 6, 4, 4, 1, 1, 6,
            4, 1, 6, 6, 2, 1, 6, 2, 2, 2, 6, 6, 4, 5, 7, 4, 4, 4, 2, 5, 1, 1,
            6, 5, 6, 4, 0, 5, 6, 4, 6, 4, 5, 1, 1, 6, 4, 4, 3, 6, 5, 6, 4, 5,
            1, 1, 3, 6, 5, 4, 6, 6, 5, 2, 5, 1, 6, 6, 5, 6, 2, 5, 6, 1, 6, 6,
            4, 6, 6, 3, 4, 3, 4, 3, 6, 4, 4, 6, 4, 6, 5, 6, 4, 5, 1, 5, 6, 1,
            5, 6, 7, 6, 1, 3, 7, 7, 4, 5, 5, 7, 5, 3, 6, 2, 5, 6, 5, 0, 6, 1,
            5, 3, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 5, 1, 2, 6, 0, 5, 6, 1, 3, 4,
            4, 4, 1, 6, 6, 3, 6, 3, 5, 1, 5, 4, 6, 1, 2, 6, 2, 4, 6, 1, 1, 6,
            6, 1, 1, 1, 4, 2, 2, 1, 1, 5, 6, 1, 4, 4, 7, 3, 4, 1, 1, 1, 1, 4,
            1, 6, 1, 6, 6, 6, 1, 6, 6, 4, 1, 0, 5, 2, 1, 4, 6, 4, 6, 2, 6, 6,
            6, 5, 7, 1, 3, 6, 7, 1, 3, 1, 7, 6, 6, 1, 6, 1, 5, 0, 4, 4, 1, 1,
            6, 5, 1, 4, 4, 6, 4, 0, 4, 5, 6, 5, 1, 2, 4, 4, 1, 0, 0, 4, 1, 1,
            5, 4, 3, 1, 5, 1, 5, 6, 0, 1, 0, 6, 3, 4, 1, 3, 4, 5, 1, 1, 6, 1,
            1, 5, 5, 4, 1, 4, 4, 1, 1, 4, 6, 1, 7, 4, 1, 6, 5, 4, 6, 4, 1, 1,
            6, 3, 6, 2, 1, 7, 4, 4, 7, 1, 6, 7, 0, 5, 1, 7, 7, 1, 1, 6, 1, 3,
            1, 4, 1, 0, 6, 1, 1, 1, 6, 5, 7, 5, 4, 1, 5, 0, 5, 4, 4, 6, 1, 3,
            4, 6, 3, 1, 3, 1, 5, 3, 6, 1, 3, 3, 4, 1, 4, 5, 6, 6, 4, 6, 4, 4,
            4, 5, 1, 6, 1, 3, 4, 4, 1, 1, 6, 6, 5, 5, 6, 5, 1, 7, 7, 1, 6, 5,
            6, 4, 0, 4, 2, 4, 3, 6, 6, 5, 3, 6, 1, 4, 6, 6, 6, 6, 1, 6, 6, 6,
            4, 6, 1, 7, 6, 6, 4, 6, 4, 6, 6, 6, 3, 1, 6, 6, 5, 1, 4, 1, 4, 1,
            0, 6, 7, 7, 1, 1, 1, 5, 1, 1, 6, 6, 1, 6, 6, 6, 2, 1, 2, 1, 4, 6,
            3, 4, 4, 6, 1, 4, 4, 6, 6, 6, 1, 1, 6, 7, 6, 6, 3, 7, 3, 1, 6, 1,
            4, 4, 6, 4, 6, 6, 6, 4, 1, 6, 4, 1, 4, 6]
        songs['cluster']  = cluster_lab
        if cluster == 'Cluster1':
            sample = songs[songs.cluster == 0]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)
            
            

            #Para cambiar a un mapa claro cambiar el paramétro tiles a "cartodbpositron"
            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            # ax = sns.lmplot(x="energy", y="danceability", hue="cluster", data=songs)
            # ax.set(ylabel = "Crímenes")
            # ax.set(xlabel = "Dellito")

            # st.pyplot(ax)

            # st.image('output.png', use_column_width=True)
            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 0].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)

            
            # st.write(sample)
        if cluster == 'Cluster2':
            sample = songs[songs.cluster == 1]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            sample = songs[songs.cluster == 1].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster3':
            sample = songs[songs.cluster == 2]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 2].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster4':
            sample = songs[songs.cluster == 3]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 3].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster5':
            sample = songs[songs.cluster == 4]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 4].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)

        if cluster == 'Cluster6':
            sample = songs[songs.cluster == 5]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 5].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster7':
            sample = songs[songs.cluster == 6]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 6].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster8':
            sample = songs[songs.cluster == 7]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 7].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
    

    if songs_from == 'Rafa Songs':
        cluster = st.sidebar.radio('Select', ('Cluster1','Cluster2','Cluster3','Cluster4','Cluster5','Cluster6','Cluster7','Cluster8'))
        col1, col2 = st.beta_columns(2)
        songs = data[(data.label == 'R') | (data.label == 'RF') | (data.label == 'FR')]
        # print(songs)
        cluster_lab = [1, 2, 1, 3, 3, 2, 3, 1, 6, 3, 3, 3, 1, 1, 2, 1, 2, 1, 1, 1, 3, 0,
            1, 2, 2, 3, 2, 6, 1, 1, 3, 2, 1, 1, 2, 1, 3, 2, 3, 1, 2, 1, 1, 2,
            1, 1, 2, 5, 3, 1, 1, 2, 3, 2, 2, 3, 1, 2, 7, 2, 3, 1, 5, 2, 2, 7,
            1, 1, 1, 6, 1, 1, 7, 7, 2, 1, 1, 2, 1, 1, 6, 3, 3, 2, 2, 1, 2, 2,
            1, 6, 2, 1, 1, 1, 1, 2, 7, 1, 3, 3, 1, 2, 7, 6, 6, 1, 3, 2, 1, 1,
            2, 1, 3, 2, 3, 1, 1, 1, 2, 2, 7, 1, 6, 2, 2, 3, 2, 3, 1, 1, 1, 7,
            1, 1, 6, 2, 1, 6, 2, 3, 3, 6, 3, 2, 7, 3, 1, 2, 3, 2, 6, 1, 1, 6,
            2, 7, 2, 2, 7, 2, 1, 1, 1, 2, 3, 5, 3, 3, 6, 2, 2, 3, 1, 2, 2, 2,
            0, 6, 2, 3, 0, 1, 6, 6, 2, 6, 1, 2, 6, 1, 1, 5, 6, 2, 5, 2, 5, 1,
            1, 2, 6, 1, 2, 1, 6, 1, 2, 4, 4, 4, 1, 2, 6, 3, 6, 2, 1, 3, 2, 2,
            6, 4, 3, 3, 6, 1, 1, 1, 1, 5, 4, 1, 2, 2, 1, 4, 6, 6, 2, 5, 5, 1,
            1, 3, 1, 3, 1, 6, 4, 2, 1, 1, 5, 3, 1, 1, 6, 2, 1, 1, 1, 1, 7, 1,
            6, 2, 1, 1, 3, 2, 7, 2, 0, 3, 4, 4, 5, 1, 1, 5, 6, 6, 3, 3, 6, 2,
            3, 3, 2, 1, 6, 0, 0, 2, 2, 2, 1, 3, 0, 4, 7, 2, 6, 2, 2, 1, 1, 4,
            1, 6, 3, 3, 3, 2, 2, 2, 1, 2, 3, 3, 1, 2, 1, 1, 3, 6, 1, 4, 6, 1,
            3, 1, 2, 1, 3, 1, 3, 2, 3, 6, 3, 4, 3, 2, 3, 6, 5, 4, 4, 5, 5, 1,
            4, 2, 1, 1, 3, 3, 2, 1, 6, 2, 6, 3, 3, 3, 3, 0, 1, 3, 3, 1, 0, 0,
            1, 2, 4, 2, 3, 3, 2, 6, 4, 1, 3, 6, 2, 1, 1, 3, 6, 3, 6, 6, 6, 3,
            6, 6, 1, 6, 4, 6, 2, 2, 1, 2, 3, 6, 4, 3, 6, 4, 2, 6, 1, 1, 1, 6,
            4, 6, 0, 4, 3, 1, 6, 1, 4, 5, 4, 3, 6, 6, 4, 6, 1, 6, 2, 2, 6, 4,
            7, 3, 4, 3, 6, 6, 3, 6, 5, 3, 3, 6, 6, 3, 6, 3, 2, 3, 6, 5, 6, 3,
            3, 1, 6, 1, 1, 6, 5, 3, 2, 2, 2, 5, 2, 1, 3, 1, 3, 4, 6, 7, 4, 4,
            3, 6, 3, 3, 6, 2, 7, 3, 3, 1, 1, 1, 6, 2, 4, 6, 2, 6, 1, 4, 1, 5,
            4, 3, 1, 2, 3, 2, 3, 6, 1, 1, 6, 1, 3, 1, 3, 6, 5, 3, 7, 3, 4, 6,
            4, 3, 3, 6, 3, 4, 1, 3, 3, 1, 1, 1, 3, 3, 4, 3, 3, 2, 1, 3, 6, 4,
            3, 3, 1, 7, 5, 3, 6, 1, 6, 3, 1, 6, 2, 6, 6, 3, 3, 4, 6, 3, 6, 4,
            5, 0, 4, 4, 2, 3, 1, 6, 6, 3, 1, 2, 6, 1, 5, 2, 1, 1, 5, 6, 2, 1,
            1, 3, 1, 4, 1, 3, 3, 1, 2, 6, 5, 3, 7, 2, 3, 0, 4, 2, 3, 2, 3, 3,
            2, 6, 3, 1, 1, 3, 3, 3, 3, 5, 1, 3, 6, 2, 4, 3, 3, 6, 6, 1, 1, 1,
            2, 3, 2, 3, 4, 6, 2, 7, 1, 3, 4, 6, 1, 1, 1, 1, 4, 6, 4, 2, 6, 1,
            6, 1, 2, 5, 1, 2, 3, 3, 2, 1, 2, 1, 6, 2, 6, 1, 6, 3, 3, 3, 1, 3,
            6, 6, 1, 5, 6, 1, 2, 3, 2, 2, 1, 3, 1, 3, 1, 2, 4, 1, 5, 1, 3, 3,
            1, 6, 6, 6]
        songs['cluster']  = cluster_lab
        if cluster == 'Cluster1':
            sample = songs[songs.cluster == 0]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 0].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster2':
            sample = songs[songs.cluster == 1]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 1].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster3':
            sample = songs[songs.cluster == 2]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 2].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster4':
            sample = songs[songs.cluster == 3]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 3].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster5':
            sample = songs[songs.cluster == 4]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 4].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster6':
            sample = songs[songs.cluster == 5]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 5].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster7':
            sample = songs[songs.cluster == 6]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 6].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster8':
            sample = songs[songs.cluster == 7]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 7].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
            
    if songs_from == 'Both Songs':
        cluster = st.sidebar.radio('Select', ('Cluster1','Cluster2','Cluster3','Cluster4','Cluster5','Cluster6','Cluster7'))
        col1, col2 = st.beta_columns(2)
        songs = data[(data.label == 'RF') | (data.label == 'FR')]
        # print(songs)
        cluster_lab = [3, 1, 3, 5, 6, 1, 6, 3, 3, 6, 5, 5, 3, 6, 1, 3, 5, 6, 6, 6, 3, 0,
            3, 1, 5, 6, 6, 6, 3, 6, 5, 2, 6, 3, 1, 5, 5, 1, 6, 3, 5, 5, 3, 1,
            5, 6, 1, 2, 6, 3, 3, 1, 5, 1, 1, 6, 3, 1, 4, 5, 5, 3, 2, 1, 1, 4,
            5, 3, 6, 6, 3, 5, 4, 4, 5, 6, 6, 5, 3, 3, 3, 6, 5, 5, 1, 3, 6, 5,
            3, 3, 1, 3, 6, 3, 3, 5, 4, 6, 6, 5, 3, 6, 4, 1, 6, 3, 5, 1, 3, 3,
            1, 6, 5, 5, 6, 6, 3, 3, 1, 5, 4, 3, 6, 1, 5, 6, 1, 5, 6, 3, 3, 4,
            3, 3, 6, 5, 3, 3, 1, 6, 5, 3, 3, 1, 4, 6, 6, 1, 5, 1, 6, 3, 3, 6,
            5, 4, 3, 5, 4, 1, 5, 3, 6, 1, 6, 2, 6, 6, 3, 5, 1, 6, 5, 1, 5, 5,
            0, 3, 5, 5, 0, 2, 6, 6, 1, 6, 3, 6, 6, 5, 6, 2, 3, 5, 2, 5, 2, 5,
            3, 5, 3, 3, 1, 6, 6, 3, 1, 3, 3, 6, 3, 1, 1, 1, 5, 6, 3, 6, 1, 5,
            1, 5, 3, 3, 2, 6, 3, 3, 6, 6, 2, 2, 6, 3, 6, 1, 1, 0, 5, 4, 1, 1,
            3, 5, 5, 1, 6, 6, 1, 3, 3, 3, 3, 6, 6, 3, 5, 1, 5, 6, 2, 3, 3, 1,
            3, 1, 5, 3, 5, 1, 6, 5, 3, 3, 3, 1, 6, 3, 6, 0, 3, 2, 2, 6, 6, 6,
            6, 6, 6, 5, 5, 6, 3, 3, 6, 4, 6, 4, 6, 1, 3, 1, 1, 5, 3, 6, 6, 2,
            5, 5, 3, 5, 6, 5, 2, 5, 3, 1, 2, 1, 6, 5, 5, 5, 5, 3, 5, 5, 3, 5,
            3, 6, 3, 1, 6, 5, 3]
        songs['cluster']  = cluster_lab
        if cluster == 'Cluster1':
            sample = songs[songs.cluster == 0]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 0].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster2':
            sample = songs[songs.cluster == 1]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 1].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster3':
            sample = songs[songs.cluster == 2]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 2].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster4':
            sample = songs[songs.cluster == 3]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 3].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster5':
            sample = songs[songs.cluster == 4]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 4].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster6':
            sample = songs[songs.cluster == 5]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 5].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        if cluster == 'Cluster7':
            sample = songs[songs.cluster == 6]
            col1.markdown('### Danceability')
            m = 'mean: ' + str(sample['danceability'].mean())
            s = 'standard deviation: ' + str(sample['danceability'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Energy')
            m = 'mean: ' + str(sample['energy'].mean())
            s = 'standard deviation: ' + str(sample['energy'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Key')
            m = 'mean: ' + str(sample['key'].mean())
            s = 'standard deviation: ' + str(sample['key'].std())
            col1.markdown(m)
            col1.markdown(s)
            col1.markdown('### Loudness')
            m = 'mean: ' + str(sample['loudness'].mean())
            s = 'standard deviation: ' + str(sample['loudness'].std())
            col1.markdown(m)
            col1.markdown(s)
            col2.markdown('### Mode')
            m = 'mean: ' + str(sample['mode'].mean())
            s = 'standard deviation: ' + str(sample['mode'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Valence')
            m = 'mean: ' + str(sample['valence'].mean())
            s = 'standard deviation: ' + str(sample['valence'].std())
            col2.markdown(m)
            col2.markdown(s)
       
            col2.markdown('### Tempo')
            m = 'mean: ' + str(sample['tempo'].mean())
            s = 'standard deviation: ' + str(sample['tempo'].std())
            col2.markdown(m)
            col2.markdown(s)
            col2.markdown('### Country')
            m = 'mode: ' + str(stat.mode(sample.country))
            col2.markdown(m)
            col2.markdown('### Languge')
            m = 'mode: ' + str(stat.mode(sample.song_language))
            col2.markdown(m)

            heat = sample[['lat','lon']]
            heat.lat.fillna(0,inplace=True)
            heat.lon.fillna(0,inplace=True)
            m6=folium.Map(location=[0, 0],tiles='cartodbdark_matter',zoom_start=1)
            HeatMap(data=heat,radius=20).add_to(m6)
            folium_static(m6)

            st.sidebar.markdown('#### Random song from selected cluster: ')
            sample = songs[songs.cluster == 6].sample()
            st.sidebar.markdown(str(sample.iat[0,18]))
            for artist in sample['artist(s)']:
                st.sidebar.markdown(artist)
        

            
            
        
        
        
