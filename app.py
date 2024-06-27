import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util

# Load the dataset
dataset_path = 'plas_tech_dataset (1).json'
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
def chatbot_response(user_input, conversation_history):
    # Combine the conversation history with the new user input
    combined_input = ' '.join(conversation_history + [user_input])
    combined_input = preprocess_text(combined_input)
    user_input_embedding = model.encode(combined_input, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(user_input_embedding, context_embeddings)[0]
    
    # Find the index of the most similar context
    most_similar_idx = cosine_scores.argmax().item()
    
    return df['response'].iloc[most_similar_idx]

# Streamlit App
st.title("Plas-tech Chatbot")
st.write("Welcome to the Plas-tech Chatbot! Type your question below and press Enter.")

# Initialize or load the conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

user_input = st.text_input("You: ")

if user_input:
    response = chatbot_response(user_input, st.session_state.conversation_history)
    st.session_state.conversation_history.append(user_input)
    st.session_state.conversation_history.append(response)
    st.write(f"Chatbot: {response}")

# Display conversation history
st.write("### Conversation History")
for i, msg in enumerate(st.session_state.conversation_history):
    if i % 2 == 0:
        st.write(f"**You:** {msg}")
    else:
        st.write(f"**Chatbot:** {msg}")
