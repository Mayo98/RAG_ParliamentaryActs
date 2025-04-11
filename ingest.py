import os
from pymongo import MongoClient
import pickle
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import fitz  # PyMuPDF
import re
import numpy as np
#from langchain.vectorstores import Chroma
import sys
import os
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Ora puoi importare e usare ChromaDB



MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "atti_parlamentari"
COLLECTION_NAME = "atti"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

#FOLDER PDF DELLE PROPOSTE DI LEGGE 
FOLDER_PATH = "PDF_LEGGI"

# Configurazione del modello di embedding
embeddings_model = HuggingFaceEmbeddings(model_name = EMBEDDINGS_MODEL_NAME)

# Funzione per estrarre testo dai PDF
def extract_text_from_pdf_Resumed(pdf_path): 
    text = ""
    doc = fitz.open(pdf_path)
    print(doc.metadata)
    for page in doc:
        page_text = page.get_text("text")
        text += page_text + "\n"

    #pattern = r"!\s*.*?__"
    #text = re.sub(pattern, "__", text, flags=re.DOTALL)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Unisce parole sillabate
    text = re.sub(r'\n+', ' ', text).strip()
    return text

def load_pdfs_from_folder(folder_path):
    pdf_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf_Resumed(pdf_path)
            pdf_texts.append(text)
    return pdf_texts

# Funzione per creare e salvare il FAISS Index
def create_faiss_index(folder_path, index_path="vectorstore"):
    documents = load_pdfs_from_folder(folder_path)
    document_objects = [Document(page_content=text) for text in documents]

    #embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    docs = text_splitter.split_documents(document_objects)

    vectorstore = FAISS.from_documents(docs, embeddings_model)

    vectorstore.save_local(index_path)
    print(f"✅ FAISS Index salvato in '{index_path}'")


# Funzione per creare FAISS Index ricavando i dati dal MongoDB
def create_faiss_index_fromMongo(index_path="vectorstoreMongo"):
    document_objects = []
    documents = []
    metadata_list = []

    for doc in collection.find({}):
        # --- Recupero metadati
        metadata = {
            "atto_id": doc.get("atto_id", ""),
            "atto_label": doc.get("atto_label", ""),
            "deputato_id": doc.get("deputato_id", ""),
            "deputato_label": doc.get("deputato_label", ""),
            "gruppo_id": doc.get("gruppo_id", ""),
            "gruppo_label": doc.get("gruppo_label", ""),
            "pdf_url": doc.get("pdf_url", ""),
            "pdf_path": doc.get("pdf_path", "")
        }

        # --- Recupero e concatenazione del testo
        intro_text = doc.get("intro", {}).get("text", "")
        articoli_text = "\n\n".join(article["text"] for article in doc.get("Articoli", []) if "text" in article)
        full_text = f"{intro_text}\n\n{articoli_text}"
        documents.append(full_text)
        metadata_list.append(metadata)

        # --- Creazione oggetti Document con relativi metadati
    document_objects = [Document(page_content=text, metadata = meta) for text, meta in zip(documents, metadata_list)]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    docs = text_splitter.split_documents(document_objects)

    vectorstore = FAISS.from_documents(docs, embeddings_model)

    vectorstore.save_local(index_path)
    print(f"✅ FAISS Index salvato in '{index_path}'")



#------------ ATTIVO 
def create_chroma_index_fromMongo(index_path="chroma_db"):
    document_objects = []
    documents = []
    metadata_list = []

    for doc in collection.find({}):
        # --- Recupero metadati
        metadata = {
            "atto_id": doc.get("atto_id", ""),
            "atto_label": doc.get("atto_label", ""),
            "deputato_id": doc.get("deputato_id", ""),
            "deputato_label": doc.get("deputato_label", ""),
            "gruppo_id": doc.get("gruppo_id", ""),
            "gruppo_label": doc.get("gruppo_label", ""),
            "pdf_url": doc.get("pdf_url", ""),
            "pdf_path": doc.get("pdf_path", "")
        }

        # --- Recupero e concatenazione del testo
        intro_text = doc.get("intro", {}).get("text", "")
        articoli_text = "\n\n".join(article["text"] for article in doc.get("Articoli", []) if "text" in article)
        full_text = f"{intro_text}\n\n{articoli_text}"
        documents.append(full_text)
        metadata_list.append(metadata)

    # --- Creazione oggetti Document con relativi metadati
    document_objects = [Document(page_content=text, metadata=meta) for text, meta in zip(documents, metadata_list)]
    
    # --- Divisione in chunk per migliorare il recupero
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=80)
    docs = text_splitter.split_documents(document_objects)

    # --- Creazione del vectorstore ChromaDB
    vectorstore = Chroma.from_documents(docs, embeddings_model, persist_directory=index_path)

    print(f"✅ ChromaDB salvato in '{index_path}'")



#------------

def create_faiss_index_fromMongo2(index_path="vectorstoreMongo"):
    document_objects = []
    all_embeddings = []
    metadata_list = []

    for doc in collection.find({}):

        # --- Genera embedding separati per introduzione e articoli
        intro_text = doc.get("intro", {}).get("text", "")
        intro_embedding = np.array(embeddings_model.embed_query(intro_text))  # Se vuoto, vettore nullo
        
        articoli_embeddings = [
            np.array(embeddings_model.embed_query(article["text"]))  # Assicurati che ogni embedding sia un array NumPy
            for article in doc.get("Articoli", []) if article.get("text")
        ]

        # --- Concateno gli embeddings se le dimensioni sono le stesse
        if articoli_embeddings:
            # Controllo che tutte le dimensioni degli embeddings siano uguali
            if len(set([embedding.shape[0] for embedding in articoli_embeddings])) == 1:
                # Concatenazione degli embeddings
                full_embedding = np.concatenate([intro_embedding] + articoli_embeddings, axis=0)
            else:
                print("⚠️ Errore: Gli embeddings hanno dimensioni diverse!")
                continue  # Skip this document if the embeddings' dimensions don't match
        else:
            full_embedding = intro_embedding

        # --- Combinazione del testo (full_text)
        full_text = intro_text + "\n\n" + "\n\n".join(str(article.get("text", "")) for article in doc.get("Articoli", []))
        
        # --- Crea metadati
        metadata = {
            "atto_label": doc.get("atto_label", ""),
            "deputato_id": doc.get("deputato_id", ""),
            "deputato_label": doc.get("deputato_label", ""),
            "gruppo_label": doc.get("gruppo_label", "")
        }
        
        document_objects.append(full_text)
        all_embeddings.append(full_embedding)
        metadata_list.append(metadata)

    # --- Verifica che la lunghezza dei documenti e degli embeddings sia la stessa
    if len(document_objects) != len(all_embeddings):
        print(f"⚠️ ERRORE: Mismatch tra documenti ({len(document_objects)}) e embeddings ({len(all_embeddings)})!")
        return  

    # --- Aggiungi metadati durante la creazione dell'indice FAISS
    text_embeddings = [(doc, emb) for doc, emb in zip(document_objects, all_embeddings)]

    # --- Creazione del FAISS Index con metadati
    vectorstore = FAISS.from_embeddings(
        text_embeddings, 
        embeddings_model, 
        metadatas=metadata_list  # Aggiungi i metadati separatamente
    )

    # --- Salvataggio
    vectorstore.save_local(index_path)
    print(f"✅ FAISS Index salvato in '{index_path}'")



if __name__ == "__main__":
    folder_path = FOLDER_PATH
    create_chroma_index_fromMongo()
    #create_faiss_index_fromMongo()