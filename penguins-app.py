import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Aplikasi Prediksi Sederhana 

Aplikasi ini memprediksi dengan **Palmer Penguin**! hehehewkwk

Data yang diperoleh dari [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) di R oleh Allison Horst.
""")

st.sidebar.header('Fitur Masukan Pengguna')

st.sidebar.markdown("""
[Contoh CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Mengumpulkan fitur input pengguna ke dalam dataframe
uploaded_file = st.sidebar.file_uploader("Upload CSV MU _beta_", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Menggabungkan fitur input pengguna dengan seluruh kumpulan data penguin
# Ini akan berguna untuk fase pengkodean
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# Pengodean fitur ordinal
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Hanya memilih baris pertama (data input pengguna)

# Menampilkan fitur input pengguna
st.subheader('Input User')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Menunggu file CSV untuk diunggah. Saat ini menggunakan contoh parameter input (ditunjukkan di bawah).')
    st.write(df)

# Dibaca dalam model klasifikasi tersimpan
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Terapkan model untuk membuat prediksi
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediksi')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Probabilitas Prediksi')
st.write(prediction_proba)

#tentukan simbol
tickerSymbol = 'GOOGL'
#bisa ambil salah satu
#tickerSymbol = 'MSFT' / 'GOOGL' / 'AAPL'

#dapatkan data tentang ticker 
tickerData = yf.Ticker(tickerSymbol)
#harga historis untuk tiket
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
# Open	High	Low	Close	Volume	Dividends	Stock Splits

st.subheader('Tabel Prediksi')
st.write('P.')
#st.write("""
## Harga penutup
#""")
st.line_chart(tickerDf.Close)

st.subheader('Volume Prediksi')
st.write('V.')
#st.write("""
    ### Harga volume 
#""")
st.line_chart(tickerDf.Volume)