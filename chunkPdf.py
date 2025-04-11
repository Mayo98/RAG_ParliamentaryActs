import re
import fitz  # PyMuPDF

# Funzione per estrarre il testo dal PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    print(f"📄 Elaborazione di: {pdf_path}")

    for page in doc:
        text += page.get_text("text") + "\n"

    # Pulizia del testo: unisce parole spezzate e rimuove spazi multipli
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  
    text = re.sub(r'\n+', ' ', text).strip()  

    return text

# Funzione per suddividere il testo in base agli articoli
def split_text_into_sections(text):
    sections = []

    # Trova la parte tra le due occorrenze di "PROPOSTA DI LEGGE"
    match = re.findall(r"PROPOSTA DI LEGGE(.*?)PROPOSTA DI LEGGE", text, re.DOTALL)
    if match:
        sections.append(match[0].strip())  # Salva la parte introduttiva
        text = text.split("PROPOSTA DI LEGGE", 2)[-1]  # Mantiene il resto del testo dopo la seconda occorrenza

    # Suddivide in articoli (Art. X)
    articles = re.split(r"(Art\.\s\d+)", text)

    # Ricostruisce i blocchi per mantenere il titolo degli articoli
    for i in range(1, len(articles), 2):  # Itera saltando ogni due (Articolo + Contenuto)
        article_title = articles[i].strip()
        article_text = articles[i+1].strip() if i+1 < len(articles) else ""
        sections.append(f"{article_title} {article_text}")  # Combina titolo e contenuto

    return sections

# Funzione principale per processare il PDF
def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    sections = split_text_into_sections(text)
    
    # Stampa le sezioni estratte
    for idx, section in enumerate(sections):
        print(f"\n🔹 Sezione {idx + 1}:\n{section}")  # Mostra solo i primi 500 caratteri

    return sections

# Esegui lo script su un PDF
if __name__ == "__main__":
    pdf_path = "pdf_leggi/19PDL0004660.pdf"  # Inserisci il path del tuo PDF
    sections = process_pdf(pdf_path)

    
