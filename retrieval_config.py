import os
import PyPDF2
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM 
from sentence_transformers import SentenceTransformer  
from langchain_community.vectorstores import FAISS
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from scipy.spatial import distance
import pickle
import pymongo
from pymongo import MongoClient
import sys
import os
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
model_name = os.environ.get("MODEL", "llama3:latest")  #llama3:8b-instruct-q8 0  

# Connessione a MongoDB
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "atti_parlamentari"
COLLECTION_NAME = "atti"

client = pymongo.MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]


#INDEX_PATH_LOCAL = "vectorstoreMongo"
INDEX_PATH_LOCAL = "chroma_db"


# Percorso vectorstore da Flask
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Prende la cartella di retrieval_config.py
#BASE_DIR = os.path.join(BASE_DIR, "..")  # Sale di un livello
BASE_DIR = os.path.abspath(BASE_DIR)  # Risolve il percorso assoluto

INDEX_PATH = os.path.join(BASE_DIR, "chroma_db")

#FAISS
def load_faiss_index(index_path=INDEX_PATH):
    print(f"Caricamento FAISS da: {index_path} -- base dir: {BASE_DIR}")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    #index_path = os.path.abspath("vectorstoreMongo")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

#CHROMA
def load_chroma_index(index_path=INDEX_PATH):
    print(f"Caricamento Chroma da: {index_path} -- base dir: {BASE_DIR}")
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # Carica il vectorstore Chroma
    return Chroma(persist_directory=index_path, embedding_function=embeddings)


#CARICO VECTORSTORE vectorstore = load_faiss_index()

vectorstore = load_chroma_index()



PROMPT_TEMPLATE = """Utilizza i seguenti elementi di contesto per rispondere alla domanda. Leggi attentamente tutto il contesto prima di rispondere. 
    Genera la risposta ricavando elementi solo dal contesto.
    Se non conosci la risposta, dì semplicemente che non la sai.
    Rispondi in modo breve. Rispondi con linguaggio giuridico.
    Se citi degli articoli nella risposta, fornisci sempre il titolo della legge che contiene gli articoli che stai citando
    Se la query richiede le opinioni di una persona, considera il contesto come pensieri di quella persona in particolare.
    Se la query richiede se sono state presentate proposte di legge o atti riguardanti il contesto, rispondi con massimo 2 frasi se il contesto presenta quanto richiesto, altrimenti rispondi che non sono state presentate.
    Ogni Articolo può presentare dei commi, solitamente identificati da un numero. Considerali sempre come parte integrante dell'articolo a cui fanno riferiemnto.
    Tratta ogni articolo e i propri comma a se stanti.
    Se lavori su più proposte di legge, considerali tutti prima di produrre la risposta, leggendo attentamente tutto il contenuto. 
    Nella risposta cita sempre l'autore e il partito politico fornito nel contesto che leggi.
    Mantieni la risposta affine al contesto.
    Rispondi sempre in Italiano. 
    
    Il contesto è il seguente: 
    {context}
    Domanda: {query}
    Risposta:"""
PROMPTS = {
    "deputato": 
    """Utilizza i seguenti elementi di contesto per rispondere alla domanda. Leggi attentamente tutto il contesto prima di rispondere. 
    Genera la risposta ricavando elementi solo dal contesto, se il contesto non presenta informazioni rilevanti alla query non usare la tua conoscienza interna per rispondere.
    Se non conosci la risposta, dì semplicemente che non la sai.
    Rispondi in modo breve, massimo 4 frasi. Rispondi con linguaggio giuridico.
    Il contesto fornisce proposte di legge presentate da deputati.
    Se citi degli articoli nella risposta, fornisci sempre il titolo della legge che contiene gli articoli che stai citando.
    Se la query chiede delle opinioni, considera il contesto come opinioni di un deputato parlamentare.
    Se la query richiede se sono state presentate proposte di legge o atti riguardanti il contesto, rispondi con massimo due frasi informative se nel contesto trovi riferimenti a quanto chiesto, altrimenti rispondi che non sono state presentate.
    Ogni Articolo può presentare dei commi, solitamente identificati da un numero. Considerali sempre come parte integrante dell'articolo a cui fanno riferiemnto.
    Tratta ogni articolo e i propri comma a se stanti.
    Se nel conteto ci sono più proposte di legge, considerale tutte prima di produrre la risposta, leggendo attentamente tutto il contenuto. 
    Nella risposta cita sempre l'autore e il partito politico fornito nel contesto che leggi.
    Mantieni la risposta affine al contesto.
    Rispondi sempre in Italiano. 
    
    Il contesto è il seguente: 
    {context}
    Domanda: {query}
    Risposta:""",

    "gruppo": 
    """Utilizza i seguenti elementi di contesto per rispondere alla domanda. Leggi attentamente tutto il contesto prima di rispondere. 
    Genera la risposta ricavando elementi solo dal contesto, se il contesto non presenta informazioni rilevanti alla query non usare la tua conoscienza interna per rispondere.
    Se non conosci la risposta, dì semplicemente che non la sai.
    Rispondi in modo breve, massimo 3 frasi. Rispondi con linguaggio giuridico.
    Se citi degli articoli nella risposta, fornisci sempre il titolo della legge che contiene gli articoli che stai citando
    Se la query richiede le opinioni su un certo argomento, considera il contesto come opinioni di un gruppo politico.
    Se la query richiede se sono state presentate proposte di legge o atti riguardanti il contesto, rispondi in modo breve ed elenca le proposte di legge che hai trovato se nel contesto trovi riferimenti a quanto chiesto.
    Se non trovi proposte di legge pertinenti, rispondi che non sono state presentate proposte di legge in merito a quanto chiesto.
    Ogni Articolo può presentare dei commi, solitamente identificati da un numero. Considerali sempre come parte integrante dell'articolo a cui fanno riferiemnto.
    Tratta ogni articolo e i propri comma a se stanti.
    Se nel contesto vi sono più proposte di legge, considerale tutte prima di produrre la risposta. 
    Nella risposta cita sempre l'autore e il partito politico fornito nel contesto che leggi.
    Mantieni la risposta affine al contesto.
    Rispondi sempre in Italiano. 
    
    Il contesto è il seguente: 
    {context}
    Domanda: {query}
    Risposta:""",

    "uriLegge": 
    """Utilizza i seguenti elementi di contesto per rispondere alla domanda. Leggi attentamente tutto il contesto prima di rispondere. 
    Genera la risposta ricavando elementi solo dal contesto.
    Se non conosci la risposta, dì semplicemente che non la sai.
    Se citi degli articoli nella risposta, fornisci sempre il titolo della legge che contiene gli articoli che stai citando
    Se la query richiede le opinioni di una persona, considera il contesto come pensieri di quella persona in particolare.
    Ogni Articolo può presentare dei commi, solitamente identificati da un numero. Considerali sempre come parte integrante dell'articolo a cui fanno riferiemnto.
    Tratta ogni articolo e i propri comma a se stanti.
    Se lavori su più proposte di legge, considerali tutti prima di produrre la risposta, leggendo attentamente tutto il contenuto. 
    Nella risposta cita sempre l'autore e il partito politico fornito nel contesto che leggi.
    Mantieni la risposta affine al contesto.
    Rispondi sempre in Italiano. 
    
    Il contesto è il seguente: 
    {context}
    Domanda: {query}
    Risposta:""",

    "default": "..."
}
def get_vectorstore():
    return vectorstore
def get_model_name():
    return model_name
def get_prompt():
    return PROMPT_TEMPLATE
def get_prompt(option_key):
    return PROMPTS.get(option_key, PROMPTS["default"])