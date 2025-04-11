import os
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from pymongo import MongoClient
import html 

#  CONFIGURAZIONE MONGO DB
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "atti_parlamentari"
COLLECTION_NAME = "atti"

# Connessione a MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


DOWNLOAD_FOLDER = "PDF_LEGGI"

collection.delete_many({})

#  FUNZIONE PER SCARICARE I PDF
def download_pdf(url, download_folder):
    try:
        pdf_name = url.split("/")[-1]
        pdf_path = os.path.join(download_folder, pdf_name)
        
        response = requests.get(url)
        response.raise_for_status()
        
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        print(f"PDF scaricato: {pdf_name}")
        
        return pdf_path  # Restituisce il percorso del file salvato
    except Exception as e:
        print(f" Errore nel download di {url}: {e}")
        return None


sparql_endpoint = "https://dati.camera.it/sparql"


query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>

SELECT DISTINCT ?atto_1 ?atto_1_label ?deputato_2 ?deputato_2_label ?gruppoParlamentare_8 ?gruppoParlamentare_8_label ?dc_relation WHERE {
  ?atto_1 rdf:type <http://dati.camera.it/ocd/atto>;
          rdfs:label ?atto_1_label;
          <http://dati.camera.it/ocd/primo_firmatario> ?deputato_2;
          dc:relation ?dc_relation;
          <http://dati.camera.it/ocd/rif_leg> <http://dati.camera.it/ocd/legislatura.rdf/repubblica_19>.
  
  ?deputato_2 rdf:type <http://dati.camera.it/ocd/deputato>;
              rdfs:label ?deputato_2_label;
              (<http://dati.camera.it/ocd/aderisce>/<http://dati.camera.it/ocd/rif_gruppoParlamentare>) ?gruppoParlamentare_8.

  ?gruppoParlamentare_8 rdf:type <http://dati.camera.it/ocd/gruppoParlamentare>;
                        rdfs:label ?gruppoParlamentare_8_label.

  FILTER(CONTAINS(STR(?dc_relation), ".pdf"))
}
"""

#QUERY SPARQL
sparql = SPARQLWrapper(sparql_endpoint)
sparql.setQuery(query)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

#  CARTELLA PDF
download_folder = DOWNLOAD_FOLDER
os.makedirs(download_folder, exist_ok=True)

# SALVATAGGIO
for result in results["results"]["bindings"]:
    atto_id = result["atto_1"]["value"]
    #atto_label = result["atto_1_label"]["value"]
    atto_label = html.unescape(result["atto_1_label"]["value"])
    deputato_id = result["deputato_2"]["value"]
    deputato_label = result["deputato_2_label"]["value"]
    gruppo_id = result["gruppoParlamentare_8"]["value"]
    gruppo_label = result["gruppoParlamentare_8_label"]["value"]
    pdf_url = result["dc_relation"]["value"]

    # Scarica il PDF
    pdf_path = download_pdf(pdf_url, download_folder)

    # Crea il documento JSON
    document = {
        "atto_id": atto_id,
        "atto_label": atto_label,
        "deputato_id": deputato_id,
        "deputato_label": deputato_label,
        "gruppo_id": gruppo_id,
        "gruppo_label": gruppo_label,
        "pdf_url": pdf_url,
        "pdf_path": pdf_path if pdf_path else None
    }

    # Inserisci in MongoDB
    collection.insert_one(document)
    print(f" Inserito in MongoDB: {atto_label}")

print(" Completato! Tutti i dati sono stati salvati in MongoDB.")
