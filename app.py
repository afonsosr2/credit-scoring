import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import DropFeatures, OneHotEncodingNames, OrdinalFeature, MinMaxWithFeatNames
from sklearn.pipeline import Pipeline
import joblib
from joblib import load

dados = pd.read_csv("https://raw.githubusercontent.com/afonsosr2/credit-scoring/main/df_clean.csv")

# Criando o formulário de cadastro dentro do aplicativo do streamlit ###

st.write("# Simulador de Avaliação de Crédito")

st.write("### Idade")
input_idade = float(st.slider("Selecione a sua idade", 18, 100))

st.write("### Nível de Escolaridade")
input_grau_escolaridade = st.selectbox("Qual é o seu grau de escolaridade?", dados['Grau_escolaridade'].unique())

st.write("### Estado Civil")
input_estado_civil = st.selectbox("Qual é o seu estado civil?", dados['Estado_civil'].unique())

st.write("### Família")
input_membros_familia = float(st.slider("Selecione quantos membros tem a sua família?", 1, 20))

st.write("### Carro próprio")
input_carro_proprio = st.radio("Você possui um automóvel?", ["Sim", "Não"])
input_carro_proprio_dict = {"Sim": 1, "Não": 0}
input_carro_proprio = input_carro_proprio_dict.get(input_carro_proprio)

st.write("### Casa própria")
input_casa_propria = st.radio("Você possui imóvel próprio?", ["Sim", "Não"])
input_casa_propria_dict = {"Sim": 1, "Não": 0}
input_casa_propria = input_casa_propria_dict.get(input_casa_propria)

st.write("### Tipo de moradia")
input_tipo_moradia = st.selectbox("Qual é o seu tipo de moradia?", dados['Moradia'].unique())

st.write("### Categoria de renda")
input_categoria_renda = st.selectbox("Qual é a sua categoria de renda?", dados['Categoria_de_renda'].unique())

st.write("### Ocupação")
input_ocupacao = st.selectbox("Qual é a sua ocupação?", dados['Ocupacao'].unique())

st.write("### Experiencia")
input_tempo_experiencia = float(st.slider("Qual é o seu tempo de experiência?", 0, 50))

st.write("### Rendimentos")
input_rendimentos = float(st.number_input("Digite o seu rendimento anual (em reais) e pressione Enter para confirmar", 0))

st.write("### Telefone corporativo")
input_tel_trabalho = st.radio("Você possui um telefone corporativo?", ["Sim", "Não"])
input_tel_trabalho_dict = {"Sim": 1, "Não": 0}
input_tel_trabalho = input_tel_trabalho_dict.get(input_tel_trabalho)

st.write("### Telefone fixo")
input_tel_fixo = st.radio("Você possui um telefone fixo?", ["Sim", "Não"])
input_tel_fixo_dict = {"Sim": 1, "Não": 0}
input_tel_fixo = input_tel_fixo_dict.get(input_tel_fixo)

st.write("### Email")
input_email = st.radio("Você possui um e-mail?", ["Sim", "Não"])
input_email_dict = {"Sim": 1, "Não": 0}
input_email = input_email_dict.get(input_email)

# Fim do formulário de cadastro dentro do aplicativo do streamlit ###

# lista dos dados requisitados
novo_cliente = [0,
                input_carro_proprio,
                input_casa_propria,
                input_tel_trabalho,
                input_tel_fixo,
                input_email,
                input_membros_familia,
                input_rendimentos,
                input_idade,
                input_tempo_experiencia,
                input_categoria_renda,
                input_grau_escolaridade,
                input_estado_civil,
                input_tipo_moradia,
                input_ocupacao,
                0]


# divisão dos dados de teste e treino
def data_split(df, test_size):
    SEED = 1561651
    treino_df, teste_df = train_test_split(df, test_size = test_size, random_state = SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

treino_df, teste_df = data_split(dados, 0.2)

cliente_predict_df = pd.DataFrame([novo_cliente], columns = teste_df.columns)

teste_novo_cliente = pd.concat([teste_df, cliente_predict_df], ignore_index=True)

#Pipeline
def pipeline_teste(df):

    pipeline = Pipeline([
        ('feature_dropper', DropFeatures()),
        ('OneHotEncoding', OneHotEncodingNames()),
        ('ordinal_feature', OrdinalFeature()),
        ('min_max_scaler', MinMaxWithFeatNames()),
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

#Aplicando a pipeline
teste_novo_cliente = pipeline_teste(teste_novo_cliente)

#retirando a coluna target
cliente_pred = teste_novo_cliente.drop(['Mau'], axis=1)

#Predições 
if st.button('Enviar'):
    model = joblib.load('modelo/xgb.joblib')
    final_pred = model.predict(cliente_pred)
    if final_pred[-1] == 0:
        st.success('### Parabéns! Você teve o cartão de crédito aprovado')
        st.balloons()
    else:
        st.error('### Infelizmente, não podemos liberar crédito para você agora!')