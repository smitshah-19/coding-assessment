#import dependencies
import openai  
import urllib.request
import xml.etree.ElementTree as ET
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

 # Set your OpenAI API key
openai.api_key = os.getenv("API_KEY")

def fetch_papers():
    """Fetches papers from the arXiv API and returns them as a list of strings."""
    url = 'http://export.arxiv.org/api/query?search_query=ti:llama&start=0&max_results=70'
    response = urllib.request.urlopen(url)
    data = response.read().decode('utf-8')
    root = ET.fromstring(data)

    papers_list = []

    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        paper_info = f"Title: {title}\nSummary: {summary}\n"
        papers_list.append(paper_info)

    return papers_list

# Fetch papers
papers_list = fetch_papers()

# Load pre-trained model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Create embeddings for each paper
paper_embeddings = model.encode(papers_list)

# Convert the list of embeddings to a 2D tensor
paper_embeddings_tensor = torch.tensor(np.array(paper_embeddings), dtype=torch.float32)

# Define a function for answering questions using a QA model
def answer_question(question):
    qa_model = pipeline("question-answering", model="yiyanghkust/roberta-base-squad2", tokenizer="yiyanghkust/roberta-base-squad2")
    result = qa_model({
        'question': question,
        'context': ' '.join(papers_list)
    })
    return result['answer']

# Continuously ask questions until the user decides to exit

while True:
    user_question = input("Ask a question (or type 'exit' to end): ")
    
    if user_question.lower() == 'exit':
        print("Exiting the QA chatbot.")
        break
    
    # Find the most similar paper using cosine similarity
    query_embedding = model.encode([user_question])[0]
    query_embedding_tensor = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)

    # Ensure the tensor shapes match for cosine similarity
    if len(query_embedding_tensor.shape) != 2 or len(paper_embeddings_tensor.shape) != 2:
        print("Error: Embeddings are not 2D tensors.")
        continue

    if query_embedding_tensor.shape[1] != paper_embeddings_tensor.shape[1]:
        print("Error: Embedding dimensions do not match.")
        continue

    similarities = F.cosine_similarity(query_embedding_tensor, paper_embeddings_tensor, dim=1)
    most_similar_paper_index = similarities.argmax().item()
    most_similar_paper = papers_list[most_similar_paper_index]

    print(f"Ans: (from the most similar paper): {most_similar_paper}\n")