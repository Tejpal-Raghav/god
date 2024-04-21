import os
import spacy
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Define functions and other code for app.py here...

# Load legal data - Cases
cases_directory = '/kaggle/input/legalai/Object_casedocs/'
cases_texts = load_legal_data(cases_directory)

# Load legal data - Statutes
statutes_directory = '/kaggle/input/legalai/Object_statutes/'
statutes_texts = load_legal_data(statutes_directory)

# Load SpaCy language model
nlp = spacy.load("en_core_web_sm")

# Preprocess and vectorize text for cases
tfidf_matrix_cases, vectorizer_cases = preprocess_text(cases_texts, nlp)

# Preprocess and vectorize text for statutes
tfidf_matrix_statutes, vectorizer_statutes = preprocess_text(statutes_texts, nlp)

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    with open("index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/generate-legal-document/")
async def generate_legal_document(user_query: str):
    # Vectorize user query
    query_vector_cases = vectorizer_cases.transform([user_query])
    query_vector_statutes = vectorizer_statutes.transform([user_query])

    # Retrieve the most relevant case and statute
    relevant_case = get_most_relevant_text(query_vector_cases, tfidf_matrix_cases, cases_texts)
    relevant_statute = get_most_relevant_text(query_vector_statutes, tfidf_matrix_statutes, statutes_texts)

    # Extract statutes from the relevant case
    doc = nlp(relevant_case)
    statutes = [ent.text for ent in doc.ents if ent.label_ == "LAW"]

    # Summarize the relevant case
    case_summary = "\n".join([sent.text for sent in doc.sents])

    # Generate Legal Document
    legal_document = f"Legal Document - User Query: {user_query}\n\n"
    legal_document += f"Case Summary:\n{case_summary}\n\n"
    legal_document += "Relevant Statute:\n"
    legal_document += f"{relevant_statute}\n"
    legal_document += "\nGuidance for the User:\n"
    legal_document += "To defend your friend in court, focus on presenting evidence that supports their actions were in self-defense.\n"
    legal_document += "Emphasize any mitigating circumstances and demonstrate their lack of intent to harm.\n"
    legal_document += "Consult with a qualified legal professional to build a strong defense strategy."

    return {"legal_document": legal_document}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
