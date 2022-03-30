#carregando as bibliotecas
from   turtle import width
from   click import style
from   numpy import empty
import pandas as pd
import streamlit as st
from   st_aggrid import AgGrid, GridOptionsBuilder
from   st_aggrid.shared import GridUpdateMode
from   minio import Minio
import joblib
import pickle
import joblib
import matplotlib.pyplot as plt
import math
import time
from   sklearn.preprocessing import MinMaxScaler
from   PIL import Image

# Variáveis de configuração
var_model = "dowloads\model.sav"
var_model_cluster = "dowloads\cluster.joblib"
var_dataset = "dowloads\dataset.csv"
var_bucket = "curated"
scaler = MinMaxScaler()


# Baixando os aquivos do Data Lake
client = Minio(
        "localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )

client.fget_object(var_bucket,"model.sav", var_model)
client.fget_object(var_bucket,"dataset.csv", var_dataset)
client.fget_object(var_bucket,"cluster.joblib", var_model_cluster)

# Carregando o modelo treinado e o conjunto de dados
loaded_model = joblib.load(var_model)
loaded_cluster = joblib.load(var_model_cluster)
dataset = pd.read_csv(var_dataset)
dataset.insert(0,"employee_id", dataset.index)
dataset.drop("turnover", axis="columns", inplace=True)

# Criando os componentes do app
st.sidebar.subheader("Dados da pessoa colaboradora:")
st.image(Image.open('..\images\image1.png'), width=120)
st.markdown('<p style="font-family:sans-serif; color:#187BCD; font-size: 42px;">People Analytics</p>', unsafe_allow_html=True)
st.markdown("Solução construída para obter dados de pessoas colaboradoras e/ou prever a saída delas da empresa.")
st.markdown("Para obter os dados, busque pelo 'employee_id' da pessoa. \
             Você pode usar os filtros na lateral da tabela para facilitar sua busca!\
             Se quiser prever a saída delas, insira os dados no meu lateral e clique em 'PREVER'.")

# Inserindo os inputs para o usuário entrar com os dados do funcionário: média como valor default 
satisfaction = st.sidebar.number_input("Satisfação", value=0)
yearsAtCompany = st.sidebar.number_input("Tempo na empresa", value=0)
evaluation = st.sidebar.number_input("Nota da última avaliação", value=0)
projectCount = st.sidebar.number_input("Número de projetos", value=0)
averageMonthlyHours = st.sidebar.number_input("Horas trabalhadas (média mensal)", value=0)
btn_predict = st.sidebar.button("PREVER")

# Funções auxiliares
def get_user_inputs():
    data = pd.DataFrame()
    data["satisfaction"] = [satisfaction]
    data["yearsAtCompany"] = [yearsAtCompany]
    data["evaluation"] = [evaluation] 
    data["projectCount"] = [projectCount]  
    data["averageMonthlyHours"] = [averageMonthlyHours]
    return data;

def show_selected_row(selection):
    st.markdown('<p style="font-family:sans-serif; color: Aqua; font-size: 3Opx;">VOCÊ SELECIONOU:</p>', unsafe_allow_html=True)
    st.dataframe(selection["selected_rows"])
    
def show_inputed_data(data):
    st.markdown('<p style="font-family:sans-serif; color: Aqua; font-size: 3Opx;">DADOS PARA PREVISÃO:</p>', unsafe_allow_html=True)
    st.dataframe(data)

def show_loading():
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.0005)
        my_bar.progress(percent_complete + 1)

def show_result_container(predict, probability):
     with st.container():
        new_title = '<p style="font-family:sans-serif; color: Aqua; font-size: 3Opx;">PREVISÃO:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.markdown(f'Pessoa colaboradora {"sairá" if predict == 1 else "não sairá"} da empresa ({int(probability[0][predict]*100)}% de confiança).')

def aggrid_interactive_table(df: pd.DataFrame):
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )
    options.configure_side_bar()
    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )
    return selection

# Função para mostrar valor selecionado na tabela
selection = aggrid_interactive_table(df=dataset)
if len(selection["selected_rows"]) != 0:
    show_selected_row(selection)

# Função para fazer predição ao clicar no botão predict
if btn_predict:
    data = get_user_inputs()
    predict = loaded_model.predict(data)
    probabability = loaded_model.predict_proba(data)
    show_loading()
    show_inputed_data(data)
    show_result_container(predict, probabability)
    