# Plas-tech Chatbot

## Overview
This repository contains the code and data to create a chatbot using the BERT model. The chatbot is designed to provide information about Plas-tech Energies Limited, a company dedicated to converting plastic waste into clean cooking gas. The chatbot uses a custom dataset closely related to the mission of Plas-tech.

## Objective
The goal of this project is to demonstrate the process of creating a chatbot using the BERT model. This involves collecting a custom dataset, preprocessing the data, fine-tuning the BERT model, building a chatbot interface, and deploying the chatbot.

## Dataset
The dataset used is a custom collection named `plas_tech_dataset (1).json`, which includes context-response pairs relevant to Plas-tech Energies Limited.

## Preprocessing
- **Clean the text data**: Remove special characters and convert text to lowercase.
- **Tokenize the text**: Prepare the text for input to the BERT model.
- **Prepare input tensors for BERT**: Convert the tokenized text into tensors suitable for BERT.

## Model
The BERT model is fine-tuned on the custom dataset to provide accurate responses based on the given context. The `sentence-transformers` library is used to handle sentence embeddings and similarity matching.

## Files in the Repository
- `app.py`: The main Streamlit application file that runs the chatbot.
- `plas_tech_dataset (1).json`: The dataset file containing context-response pairs.
- `plas_tech_chatbot.ipynb`: The Jupyter notebook used to create and fine-tune the model.
- `requirements.txt`: A list of dependencies required to run the project.
- `.gitignore`: Specifies files to be ignored by Git.
- `.gitattributes`: Specifies attributes for Git LFS.
- `README.md`: This README file.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/SamuelOnyangoOmondi/Build-a-chatbot.git
    cd Build-a-chatbot
    ```

2. **Set up a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

To run the Streamlit application locally:

```sh
streamlit run app.py
This will start a local web server and open a new tab in your default web browser to display your application.

Accessing the Deployed Chatbot
The chatbot is also deployed and accessible online. You can interact with the chatbot at the following URL:

[Plas-tech Chatbot](https://samuelonyangoomondi-build-a-chatbot-app-qqhynw.streamlit.app/)


Examples of Conversations
Below are some examples of conversations with the chatbot:

User: What is Plas-tech Energies Limited?

Chatbot: Plas-tech Energies Limited is dedicated to converting plastic waste into clean cooking gas.

User: What is the mission of Plas-tech?

Chatbot: Our mission is to revolutionize sustainable energy solutions by converting plastic waste into clean cooking gas.

Performance Metrics
The performance of the chatbot model is evaluated based on the accuracy of the responses to the context provided in the custom dataset. The fine-tuning process ensures that the model gives relevant and accurate responses.

Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.
