import os
import PyPDF2
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import ollama
#from langchain_community.llms import OllamaLLM
from sentence_transformers import SentenceTransformer  
from langchain_community.vectorstores import FAISS
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
import logging
from datetime import datetime
import sys
import os
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import fitz
import re
import spacy

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
model_name = os.environ.get("MODEL", "llama3:latest")  #llama3:8b-instruct-q8 0  
#model_name = os.environ.get("MODEL", "vaiton/minerva:latest")  #llama3:8b-instruct-q8 0  mistral:latest
models_to_test = ["llama3:latest", "mistral:latest"]

#----MODELS 
# llama3:latest: 8.0B Q4 
# mistral:latest: 7,2B Q4 
#minerva:latest: 2.9B Q8
#-----#


# Connessione a MongoDB
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "atti_parlamentari"
COLLECTION_NAME = "atti"

client = pymongo.MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]



##### FUNZIONI PER ESTRAZIONE NOME -- DA CAPIRE COSA FARCI #####
nlp = spacy.load("it_core_news_sm")  

def extract_name_from_query(query):
    llm = OllamaLLM(model=model_name)
    prompt = f"""Estrai solo il nome o cognome della persona menzionata nel seguente prompt, senza aggiungere altro.
    Il nome e il cognome di una persona sono solitamente identificati da una parola con la lettera Maiuscola iniziale: {query}"""""
    response = llm.invoke(prompt) 
    return response.strip() 

def extract_name_spacy(query):
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ == "PER":  # "PER" indica persone in SpaCy
            return ent.text
    return None
######        FINE ESTRAZIONE NOME                   #########




LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "retrieval_log.txt")
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_query(query, prompt, context, responses, metadata_filter):
    """Salva i dettagli della richiesta nel file di log."""
    log_entry = f"\n\n{'-'*80}\n"
    log_entry += f"🔎 Query: {query}\n"
    log_entry += f"📜 Prompt Originale:\n{prompt}\n"
    log_entry += f"📝 metadata_filter:\n{metadata_filter}\n"
    log_entry += f"{'-'*80}\n"

    for result in responses:
        model = result["model"]
        response_text = result["response"]
        log_entry += (
            f"\n📝 Model: {model}\n"
            f"📝 Risultato Generato:\n{response_text}\n"
            f"{'-'*40}\n"
        )
    logging.info(log_entry)
### RETRIEVAL USANDO VECTORSTORE COME RETRIEVER ###
def query_pdf_retrieval_vectorstore2(prompt, query, vectorstore, metadata_filter, model_name=model_name):
    """
    Funzione che esegue RAG tramite vectorstore FAISS, filtrando per metadati se specificati.

    Argomenti:
        prompt : Prompt base per il modello
        query (str): Query inserita da Utente.
        vectorstore (FAISS): Il vectorstore FAISS con i documenti e relativi metadati.
        model_name (str): Nome del modello di linguaggio.
        metadata_filter (dict, optional): Dizionario con i metadati su cui filtrare --> {"deputato_label": "Mario Rossi"}

    Returns:
        Risposta generata dal modello con RAG
    """

    #---RETRIEVER GENERALE SU VECTORSTORE 
    #retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    print("📌 Metadata Filter ricevuto:", metadata_filter)
    search_kwargs = {"k": 3}  #  K: recupera top-k docs 
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter  
    
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs) 
    retrieved_docs = retriever.invoke(query)
    print(f"🔍 Numero di documenti recuperati: {len(retrieved_docs)}")

    for i, doc in enumerate(retrieved_docs):
        print(f"\n📄 Documento {i+1}:")
        print(f"📝 Contenuto: {doc.page_content[:300]}...")  # Mostra solo i primi 300 caratteri
        print(f"📊 Metadati: {doc.metadata}")  # Mostra tutti i metadati
        print("-" * 50)
    context_with_metadata = ""
    for i, (doc) in enumerate(retrieved_docs):
        #completo prompt con metadati dei documenti ricavati
        deputato = doc.metadata.get("deputato_label", "Sconosciuto")
        gruppo = doc.metadata.get("gruppo_label", "Sconosciuto")
        docPath = doc.metadata.get("pdf_path", "Sconosciuto")
        
        context_with_metadata += (
            
            f"{doc.page_content}\n"
            f" L'autore del precedente documento è "
            f"{deputato}, appartenente al gruppo politico "
            f" {gruppo} "
            f"il nome del file è {docPath} \n, Se usi il testo precedente per la tua risposta, includi in fondo alla risposta generata, l'autore e il partito politico copiando i nomi correttamente. Includi sempre nella risposta il nome del file."
            f"**Esempio di risposta:**\n"
            f"Il documento è stato redatto da {deputato} appartenente al partito politico {gruppo}. "
            f"Il file è **{docPath}**." 
            f""
        )

    print(" Contesto passato al modello:\n", context_with_metadata)
    '''
    for i, doc in enumerate (retrieved_docs):
        print(f" Doc {i+1}:")
        print(f"Testo: {doc.page_content[:500]}...")
        print(f"Metadati:  {doc.metadata}")
        
        '''
    print("-"*50)

    
    after_rag_prompt = ChatPromptTemplate.from_template(prompt)
    #combined_prompt = prompt.format(context = pdf_texts)
    callbacks = [StreamingStdOutCallbackHandler()] 
    llm = OllamaLLM(model=model_name, callbacks=callbacks, temperature=0.2)
    after_rag_chain = (
        {"context": RunnablePassthrough(), "query": RunnablePassthrough()}  
        | after_rag_prompt  
        | llm  
    )
    #response = after_rag_chain.invoke({"context": context_with_metadata, "query": query})
    response =""
    # Salva i dati nel log
    log_query(query, prompt, context_with_metadata, response, metadata_filter)

    return response

    #return after_rag_chain.invoke({"context": context_with_metadata, "query": query})



######-------------- FINE FUNZIONE ------------------######


######-------------- INIZIO FUNZIONE CON CHROMA  # ------------------######

def query_pdf_retrieval_vectorstore(prompt, query, vectorstore, metadata_filter, model_name=model_name):
    """
    Funzione che esegue RAG tramite vectorstore FAISS, filtrando per metadati se specificati.

    Argomenti:
        prompt : Prompt base per il modello
        query (str): Query inserita da Utente.
        vectorstore (FAISS): Il vectorstore FAISS con i documenti e relativi metadati.
        model_name (str): Nome del modello di linguaggio.
        metadata_filter (dict, optional): Dizionario con i metadati su cui filtrare --> {"deputato_label": "Mario Rossi"}

    Returns:
        Risposta generata dal modello con RAG
    """

    #---RETRIEVER GENERALE SU VECTORSTORE 
    #retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    print("📌 Metadata Filter ricevuto:", metadata_filter)
    context_with_metadata = ""
    search_kwargs = {"k": 3}  #  K: recupera top-k docs 
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter  

    #Faccio il retrieval
    if "atto_id" not in metadata_filter:
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs) 
        retrieved_docs = retriever.invoke(query)
        print(f"🔍 Numero di documenti recuperati: {len(retrieved_docs)}")

        for i, doc in enumerate(retrieved_docs):
            print(f"\n📄 Documento {i+1}:")
            print(f"📝 Contenuto: {doc.page_content[:300]}...")  # Mostra solo i primi 300 caratteri
            print(f"📊 Metadati: {doc.metadata}")  # Mostra tutti i metadati
            print("-" * 50)
        for i, (doc) in enumerate(retrieved_docs):
            #completo prompt con metadati dei documenti ricavati
            deputato = doc.metadata.get("deputato_label", "Sconosciuto")
            gruppo = doc.metadata.get("gruppo_label", "Sconosciuto")
            docPath = doc.metadata.get("pdf_path", "Sconosciuto")
        
            context_with_metadata += (
                    
                f"{doc.page_content}\n"
                f" L'autore del contesto è: "
                f"{deputato}, appartenente al gruppo politico "
                f" {gruppo} "
                
                f""
            )

    #Altrimenti recupero il singolo documento dal MongoDB
    else:
        if query == "":
            query = "Forniscimi un riassunto dettagliato punto per punto del contesto"
        document = collection.find_one(metadata_filter)
        if document:
            testo_completo = document["intro"]["text"] + "\n\n"  # Aggiungo introduzione
            deputato = document["deputato_label"]
            gruppo = document["gruppo_label"]
            docPath = document["pdf_path"]
            for articolo in document.get("articoli", []):
                 testo_completo += f"{articolo['title']}\n{articolo['text']}\n\n"  # Aggiungo titolo e testi articolo

            print("📄 Testo completo recuperato:\n", testo_completo)
            context_with_metadata += (
                
                f"{testo_completo}\n"
                f" L'autore del contesto è:"
                f"{deputato}, appartenente al gruppo politico "
                f" {gruppo}, Rispondi in italiano."
            )
        
        
        else:
            print(" Nessun documento trovato per ", metadata_filter)
        

    print(" Contesto passato al modello:\n", context_with_metadata)
    '''
    for i, doc in enumerate (retrieved_docs):
        print(f" Doc {i+1}:")
        print(f"Testo: {doc.page_content[:500]}...")
        print(f"Metadati:  {doc.metadata}")
        
        '''
    print("-"*50)

    
    after_rag_prompt = ChatPromptTemplate.from_template(prompt)
    
    callbacks = [StreamingStdOutCallbackHandler()] 
    all_responses = []
    for model in models_to_test:
        print(f"\n Test con modello: {model}")
        llm = OllamaLLM(model=model, callbacks=callbacks, temperature=0.2)
        after_rag_chain = (
            {"context": RunnablePassthrough(), "query": RunnablePassthrough()}  
            | after_rag_prompt  
            | llm  
        )
        response = after_rag_chain.invoke({"context": context_with_metadata, "query": query})
        all_responses.append({
            "model": model,
            "response": response
        })

    # Salva i dati nel log
    #log_query(query, prompt, context_with_metadata, all_responses, metadata_filter)
    #response =""
   

    return response


def query_pdf_retrieval_mongoDB(prompt, query, model_name=model_name):
    retriever =  cerca_documenti(query)
    context = "\n\n".join([art["text"] for doc in retriever for art in doc["best_articles"]])

    after_rag_prompt = ChatPromptTemplate.from_template(prompt)
    #combined_prompt = prompt.format(context = pdf_texts)
    callbacks = [StreamingStdOutCallbackHandler()] 
    llm = OllamaLLM(model=model_name, callbacks=callbacks, temperature=0.3)
    after_rag_chain = (
        {"context": RunnablePassthrough(), "query": RunnablePassthrough()}  
        | after_rag_prompt  
        | llm  
    )
    #retrieved_docs = retriever.invoke(query)

        
    return after_rag_chain.invoke({"context": context, "query": query})

###### RICERCA PER SIMILARITA' IN MONGODB USANO DISTANZA COSENO
def cerca_documenti1(query, top_k=3, max_articles_per_doc=3):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()

    # Recupera i documenti con il campo "Articoli"
    documents = list(collection.find({"Articoli": {"$exists": True}}, {"_id": 1, "Articoli.embedding": 1}))

    results = []

    for doc in documents:
        articoli = doc.get("Articoli", [])
        article_similarities = []

        # Calcola la similarità per ogni articolo nel documento
        for article in articoli:
            article_embedding = np.array(article["embedding"])
            similarity = 1 - distance.cosine(query_embedding, article_embedding)
            article_similarities.append((article_embedding, similarity))  # Salva solo l'embedding

        article_similarities.sort(key=lambda x: x[1], reverse=True)

        # Seleziona solo gli embedding dei top articoli
        best_embeddings = [art[0] for art in article_similarities[:max_articles_per_doc]]

        max_similarity = article_similarities[0][1] if article_similarities else 0

        results.append({
            "best_embeddings": best_embeddings,  # Solo gli embedding
            "max_similarity": max_similarity
        })

    # Ordina i documenti in base alla miglior similarità trovata
    results.sort(key=lambda x: x["max_similarity"], reverse=True)

    # Restituisce solo gli embedding, ignorando max_similarity
    return [result["best_embeddings"] for result in results[:top_k]]



###### RICERCA PER SIMILARITA' IN MONGODB USANO DISTANZA COSENO
def cerca_documenti(query, top_k=3, max_articles_per_doc=3):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()

    # Recupera i documenti con il campo "Articoli"
    documents = list(collection.find({"Articoli": {"$exists": True}}, 
                                     {"_id": 1, "atto_label": 1, "gruppo_label": 1, "deputato_label": 1, 
                                      "intro": 1, "Articoli": 1}))

    results = []
    
    for doc in documents:
        articoli = doc.get("Articoli", [])
        article_similarities = []

        # Calcola la similarità per ogni articolo nel documento
        for article in articoli:
            article_embedding = np.array(article["embedding"])
            similarity = 1 - distance.cosine(query_embedding, article_embedding)
            article_similarities.append((article, similarity))

        # Ordina gli articoli per similarità decrescente
        article_similarities.sort(key=lambda x: x[1], reverse=True)

        # Seleziona i top `max_articles_per_doc`
        best_articles = [{"title": art[0]["title"], "text": art[0]["text"]} for art in article_similarities[:max_articles_per_doc]]

        max_similarity = article_similarities[0][1] if article_similarities else 0

        results.append({
            "document": doc,
            "best_articles": best_articles,  # Ora restituisce il testo invece che gli embedding
            "max_similarity": max_similarity  
        })

    # Ordina i documenti in base alla miglior similarità trovata tra i loro articoli
    results.sort(key=lambda x: x["max_similarity"], reverse=True)

    # Rimuove il campo "max_similarity" prima di restituire il risultato
    for result in results:
        del result["max_similarity"]

    return results[:top_k]


#Se non riesci a ricavare una risposta dal contesto, non usare la tua conoscenza interna.  Ignora i numeri degli articoli del contesto per la risposta.
def main():
   folder_path = "./single_pdf"  # Cartella contenente i PDF
   index_path = "vectorstoreMongo"
   query = []  
   prompt_template = """Utilizza i seguenti elementi di contesto per rispondere alla domanda. Leggi attentamente tutto il contesto prima di rispondere. 
    Genera la risposta ricavando elementi solo dal contesto.
    Se non conosci la risposta, dì semplicemente che non la sai.
    Rispondi in modo breve. Rispondi con linguaggio giuridico.
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
    Risposta:"""
   
   
   #vectorstore = load_faiss_index(index_path)
   
   print()
   print()
   metadata_filter =""

   #query = "La macellazione e il consumo delle carni di cane e di gatto sono vietate nel territorio italiano?"
   #query = "Sono state proposte misure in merito alle liste di attesa da parte della deputata Schlein?"
   #query = "Quale è l'opinione in merito alla tutela del collezionismo minore?"

   #name = extract_name_from_query(query)
   #cerca_documenti(query)
   metadata_filter={"deputato_label": "EDMONDO CIRIELLI, XIX Legislatura della Repubblica"}
   query_pdf_retrieval_vectorstore(prompt_template, query, vectorstore, metadata_filter)
   #query_pdf_retrieval_mongoDB(prompt_template, query)     
    
if __name__ == "__main__":
    main()

#Query
#cosa mi sai dire sugli articoli presenti nella proposta di legge che ti ho presentato, parlami accuratamente di tutti gli articoli presenti. Gli articoli in questione sono presenti dopo la dicitura PROPOSTA DI LEGGE. Rispondi in italiano. 

#query = "Parlami accuratamente di tutti gli articoli presenti nella proposta di legge che ti ho fornito. "
#query = "La deputata Brambilla, la quale ha firmato la proposta di legge nel contesto, è favorevole alle visite alle persone nei penitenziari da parte degli animali? Se si, a quali condizioni?"
#query = "La deputata BRAMBILLA, la quale ha firmato le proposte di legge nel contesto, cosa ne pensa del diritto al risarcimento per danni gli animali familiari?"
#query.append(input("Inserisci la tua query per LLM: ")) 
#query = "La deputata Brambilla, la quale ha firmato le proposta di legge nel contesto, di cosa parla nelle 2 proposte di legge che hai nel contesto? Mi fai un riassunto delle cose principali?"