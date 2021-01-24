import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.title("Song features Analysis of trending songs ♫")
st.sidebar.title("Song features Analysis of trending songs ♫")

st.markdown("This application is a Streamlit dashboard to analyze the song features and data of our dataset")
st.sidebar.markdown("This application is a Streamlit dashboard to analyze the song features and data of our dataset")

DATA_URL=("song-features-new.csv")
DATA_URL_RAFA=("songsrafa.csv")
DATA_URL_FER=("songsfer.csv")

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
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

    

        
        
        
        
        
