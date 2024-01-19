import streamlit as st
import pandas as pd

dados = pd.read_csv("https://raw.githubusercontent.com/afonsosr2/credit-scoring/main/df_clean.csv")
st.write("# Simulador de Avaliação de Créditos")

