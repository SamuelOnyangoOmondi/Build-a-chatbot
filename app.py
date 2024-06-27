import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util

# Load the dataset
dataset_path = 'C:\Users\Lenovo\Desktop\Plas-tech Chat Bot\Build-a-chatbot\plas_tech_dataset (1).json'
with open(dataset_path, 'r') as file:
    data = json.load(file)

# Convert the dataset into a DataFrame
df = pd.DataFrame(data)

# Data Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

# Apply preprocessing to the context and response columns
df['context'] = df['context'].apply(preprocess_text)
df['response'] = df['response'].apply(preprocess_text)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for the context sentences in the dataset
context_embeddings = model.encode(df['context'].tolist(), convert_to_tensor=True)

# Build the Chatbot Interface
def chatbot_response(user_input):
    user_input = preprocess_text(user_input)
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(user_input_embedding, context_embeddings)[0]
    
    # Find the index of the most similar context
    most_similar_idx = cosine_scores.argmax().item()
    
    return df['response'].iloc[most_similar_idx]

# Streamlit App
st.title("Plas-tech Chatbot")
st.write("Welcome to the Plas-tech Chatbot! Type your question below and press Enter.")

user_input = st.text_input("You: ")

if user_input:
    response = chatbot_response(user_input)
    st.write(f"Chatbot: {response}")
