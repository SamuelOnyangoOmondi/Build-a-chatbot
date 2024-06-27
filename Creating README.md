# Create README.md
readme_content = """
# Plas-tech Chatbot

## Overview
This repository contains the code and data to create a chatbot using the SentenceTransformer model. The chatbot is designed to provide information about Plas-tech Energies Limited.

## Dataset
The dataset used is custom-made to reflect the mission and operations of Plas-tech Energies Limited. The dataset includes context-response pairs relevant to the company.

## Model
The SentenceTransformer model is used to find the most similar context from the dataset for a given user query.

## Usage
1. Clone the repository.
2. Install the required libraries: `pip install sentence-transformers pandas torch`.
3. Run the notebook to preprocess data, compute embeddings, and build the chatbot interface.

## Example
```python
print(chatbot_response("What is Plas-tech Energies Limited?"))
print(chatbot_response("What is the mission of Plas-tech?"))
