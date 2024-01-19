import streamlit as st
import pandas as pd

dados = pd.read_csv("https://raw.githubusercontent.com/afonsosr2/credit-scoring/main/df_clean.csv")

st.write("# Simulador de Avaliação de Crédito")

st.write("### Idade")
input_idade = float(st.slider("Selecione a sua idade", 18, 100))

st.write("### Nível de Escolaridade")
input_grau_escolaridade = st.selectbox("Qual é o seu grau de escolaridade", dados['Grau_escolaridade'].unique())

st.write("### Estado Civil")
input_estado_civil = st.selectbox("Qual é o seu estado civil", dados['Estado_civil'].unique())

st.write("### Família")
input_membros_familia = float(st.slider("Selecione quantos membros tem a sua família", 1, 20))