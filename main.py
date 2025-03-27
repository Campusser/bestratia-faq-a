
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import json

app = FastAPI()

# Load FAQ data
with open("bestratia_faq_english.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [faq["question"] for faq in faq_data]
answers = [faq["answer"] for faq in faq_data]

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = model.encode(questions, convert_to_tensor=True)

class QueryRequest(BaseModel):
    q: str

@app.post("/search")
async def search_faq(request: QueryRequest):
    query_embedding = model.encode(request.q, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, question_embeddings)[0]
    best_match_idx = torch.argmax(similarities).item()
    return {
        "question": questions[best_match_idx],
        "answer": answers[best_match_idx],
        "score": float(similarities[best_match_idx])
    }

@app.get("/")
async def root():
    return {"message": "Bestratia FAQ Search API is running."}
