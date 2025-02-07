import streamlit as st
import openai
import faiss
import numpy as np
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS as LC_FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

# --- CONFIGURAÇÕES INICIAIS ---
# Insira sua API key do OpenAI
openai.api_key = st.secrets['OPENAI-APIKEY']

# Se preferir, você pode definir um objeto "clientFA" como no seu código original.
# Aqui, usarei o módulo openai diretamente.
clientFA = openai

# Lista de arquivos de texto
def ler_texto_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return f"Erro ao carregar {url}"
# Atualize os caminhos conforme sua estrutura local
lista_urls = [
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/1.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/2.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/3.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/4.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/5.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/6.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/7.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/8.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/9.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/10.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/11.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/12.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/13.txt",
    "https://raw.githubusercontent.com/Gabriel4210/PersonaFA/refs/heads/main/Textos/Faça A Sua Pergunta.txt"
]

lista_arquivos = [ler_texto_github(url) for url in lista_urls]

# --- FUNÇÕES ---
@st.cache_data(show_spinner=False)
def carregar_textos(textos_treino):
    documentos = []
    for arquivo in textos_treino:
        try:
            with open(arquivo, "r", encoding="utf-8") as f:
                documentos.append(f.read())
        except Exception as e:
            st.error(f"Erro ao ler o arquivo {arquivo}: {e}")
    texto_completo = " ".join(documentos)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    textos_divididos = text_splitter.split_text(texto_completo)
    return textos_divididos

@st.cache_resource(show_spinner=False)
def criar_banco_de_dados(textos):
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vector_store = LC_FAISS.from_texts(textos, embeddings)
    return vector_store

textos = carregar_textos(lista_arquivos)
banco_vetores = criar_banco_de_dados(textos)

def buscar_contexto(query, vector_store, k=3):
    docs = vector_store.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

def gerar_resposta(mensagem, banco_vetores):
    contexto = buscar_contexto(mensagem, banco_vetores)
    prompt = st.secrets['PROMPTSC']
    # Chama a API de completions
    try:
        completion = clientFA.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5
        )
        return completion.choices[0].message.content
    except Exception as e:
        resposta = f"Erro ao gerar resposta: {e}"
    return resposta

# --- INTERFACE STREAMLIT ---
# Interface do chatbot no Streamlit
st.set_page_config(page_title="Chatbot Flávio Augusto", page_icon="💬")

# Adicionar imagem do Flávio Augusto
st.image("flavio.jpg", width=200)

st.title("💬 Chatbot Flávio Augusto")

# Inicializar histórico da conversa
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Mostrar mensagens anteriores
for mensagem in st.session_state.mensagens:
    with st.chat_message(mensagem["role"]):
        st.markdown(mensagem["content"])

# Campo de entrada do usuário
entrada_usuario = st.chat_input("Digite sua pergunta...")

# Se o usuário enviar uma mensagem
if entrada_usuario:
    # Exibir pergunta no chat
    st.session_state.mensagens.append({"role": "user", "content": entrada_usuario})
    with st.chat_message("user"):
        st.markdown(entrada_usuario)

    # Gerar resposta
    resposta = gerar_resposta(entrada_usuario, banco_vetores)

    # Exibir resposta no chat
    st.session_state.mensagens.append({"role": "assistant", "content": resposta})
    with st.chat_message("assistant"):
        st.markdown(resposta)
