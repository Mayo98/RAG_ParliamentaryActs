import os
import re
import fitz  
import pymongo
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Configurazione MongoDB
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "atti_parlamentari"
COLLECTION_NAME = "atti"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


FOLDER_PATH = "PDF_LEGGI"
#collection.delete_many({})
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)


def extract_text_from_pdf_Resumed(pdf_path): 
    text = ""
    doc = fitz.open(pdf_path)
    print(f" Elaborazione di: {pdf_path}")
    
    for page in doc:
        text += page.get_text("text") + "\n"

    # formattazione testo
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Unisce parole spezzate
    text = re.sub(r'\n+', ' ', text).strip()  # Rimuove eccessi di spaziature

    return text

def splitTextInSections(text):
    sections = []
    articles = []

    # Trova la parte tra le due occorrenze di "PROPOSTA DI LEGGE"
    match = re.findall(r"PROPOSTA DI LEGGE(.*?)PROPOSTA DI LEGGE", text, re.DOTALL)
    if match:
        sections.append(("intro", match[0].strip()))  
        text = text.split("PROPOSTA DI LEGGE", 2)[-1]  # Mantiene il resto del testo dopo la seconda occorrenza

    # Suddivione art.
    article_parts = re.split(r"(Art\.\s\d+)", text)

    
    for i in range(1, len(article_parts), 2):  # Itera saltando ogni due (Art + Contenuto)
        article_title = article_parts[i].strip()
        article_text = article_parts[i + 1].strip() if i + 1 < len(article_parts) else ""
        articles.append({"title": article_title, "text": article_text})  

    sections.append(("Articoli", articles))  
    return sections
    
def create_faiss_index_fromMongo(index_path="vectorstoreMongo"):
    document_objects = []
    all_embeddings = []
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

        # --- Creazione oggetto Document
        document = Document(page_content=full_text, metadata=metadata)

def updateMongoTextEmbeddings(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf_Resumed(pdf_path)
            sections = splitTextInSections(text)

            if text:
                update_data = {}

                for title, content in sections:
                    if title == "intro":
                        #section_embedding = embeddings.embed_query(content)
                        #update_data["intro"] = {"text": content, "embedding": section_embedding}
                        update_data["intro"] = {"text": content}

                    elif title == "Articoli":
                        article_embeddings = []
                        for article in content:
                            article_embeddings.append({
                                "title": article["title"],
                                "text": article["text"],
                                #"embedding": embeddings.embed_query(article["text"])
                            })
                        update_data["Articoli"] = article_embeddings

                # Ricerca in Mongo e update
                result = collection.update_one(
                    {"$or": [{"pdf_path": pdf_path}, {"atto_label": filename.replace(".pdf", "")}]},
                    {"$set": update_data},
                    upsert=True  # Se il documento non esiste, lo crea
                )

                if result.matched_count > 0:
                    print(f"  Testo ed embedding aggiornati per: {filename}")
                else:
                    print(f"  Nuovo documento creato per: {filename}")

                    


if __name__ == "__main__":
    folder_path = FOLDER_PATH
    updateMongoTextEmbeddings(folder_path)
    print(" Completato! Testo ed embedding aggiornati in MongoDB.")
