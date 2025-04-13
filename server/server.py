from flask import Flask, jsonify, request
from pymongo import MongoClient
from flask_cors import CORS
from config import MONGO_URI, DB_NAME, COLLECTION_NAME
import importlib.util
import os
import sys
import os
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

# Aggiungi BASE_DIR al path di Python per consentire l'importazione di retrieval.py
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)  # Aggiungo la cartella superiore al path

# Importa config_retrieval
from retrieval_config import (
    vectorstore,
    get_prompt,
    model_name,
    client,
    collection
)


# Percorso assoluto di retrieval.py
retrieval_path = os.path.join(BASE_DIR, "retrieval.py")
print()
print()
print()
rretrieval_path = os.path.join(BASE_DIR, "server", "retrieval.py")  # Se retrieval.py è in server/
retrieval_spec = importlib.util.spec_from_file_location("retrieval", retrieval_path)
retrieval = importlib.util.module_from_spec(retrieval_spec)
retrieval_spec.loader.exec_module(retrieval)
# Recupera il vectorstore, il prompt e il nome del modello già caricati in retrieval.py
#vectorstore = retrieval.get_vectorstore()
#prompt = retrieval.get_prompt()
#model_name = retrieval.get_model_name()


# Connetti a MongoDB
client = MongoClient(MONGO_URI)

db = client[DB_NAME]
collection = db[COLLECTION_NAME]

@app.route("/get_deputati", methods=["GET"])
def get_deputati():
    """ Recupera tutti i deputati distinti dal database """
    print("Richiesta ricevuta per i deputati")
    deputati = collection.distinct("deputato_label")
    return jsonify(deputati)

@app.route("/get_gruppi", methods=["GET"])
def get_gruppi():
    """ Recupera tutti i gruppi politici distinti dal database """
    print("Richiesta ricevuta per gruppi")
    gruppi = collection.distinct("gruppo_label")
    return jsonify(gruppi)

@app.route("/query", methods=["POST"])
def query():
    """ Riceve una query dalla GUI e restituisce un risultato """
    data = request.json
    print("📩 Dati ricevuti dalla richiesta API:", data)  # <-- LOG IMPORTANTE
    option_key = data.get("optionKey")  # Identificatore della scelta
    query_text = data.get("query")  # Query dell'utente
    filter_value = data.get("filterValue")  # Valore selezionato (deputato, partito, legge)
    prompt = get_prompt(option_key)
    print()
    print()
    print()
    print()
    # Costruisco il filtraggio per metadata
    metadata_filter = None
    if option_key == "deputato":
        metadata_filter = {"deputato_label": filter_value}
    elif option_key == "gruppo":
        metadata_filter = {"gruppo_label": filter_value}
    elif option_key == "uriLegge":
        metadata_filter = {"atto_id": filter_value} 
        
    print("🔍 Metadata Filter:", metadata_filter)
    try:
        result = retrieval.query_pdf_retrieval_vectorstore(
            prompt=prompt,
            query=query_text,
            vectorstore=vectorstore,
            metadata_filter=metadata_filter,
            model_name=model_name
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"result": result})

    # Simula una risposta (da sostituire con interrogazione reale a MongoDB)
    #result = f"Opzione selezionata: {option}\nDomanda: {query}\nParametro extra: {filter}"
    
    #return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)  # Ascolta su tutte le interfacce
