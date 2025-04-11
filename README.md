# RAG_ParliamentaryActs

Application for questioning parliamentary acts using RAG and LLM. Developed as a Master's thesis in Computer Engineering at the **University of Florence**.

***Workflow***

**MONGODB**

1. wrapperSPARQLMongo.py: locally downloaded PDFs and metadata.
2. ingestMongo.py: text extraction from pdfs, formatting and sectioning. All stored into the mongoDB.

**VECTORSTORE**

3.  ingest.py: retrieves all the data stored in the mongoDB and generates a vectorstore (ChromaDB). Allows searching with metadata filters.

**RAG**

4. retrieval.py: retrieves top-k most relevant chunk from the vectorstore and pass them to LLM to generate response.
