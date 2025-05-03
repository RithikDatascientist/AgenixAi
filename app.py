from transformers import BertTokenizer, BertForSequenceClassification
import torch
import streamlit as st
from pathlib import Path

# Load models and tokenizer
intent_model_path = Path("C:/works/task_agenixai/intent_model")
intent_model = BertForSequenceClassification.from_pretrained(intent_model_path)

sentiment_model_path = Path("C:/works/task_agenixai/sentiment_model")
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_path)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Label maps
id2label_intent = {
    0: "ask_prescription",
    1: "complaint",
    2: "discuss_symptoms",
    3: "follow_up",
    4: "general_questions",
    5: "request_appointment"
}

id2label_sentiment = {
    0: "apologetic",
    1: "concerned",
    2: "formal",
    3: "friendly",
    4: "negative",
    5: "neutral",
    6: "positive",
    7: "urgent"
}

# Streamlit app
st.title("Patient Message Classifier")

user_input = st.text_input("Enter a patient message:")

if user_input:
    # Tokenize once for both models
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

    # Intent Prediction
    intent_outputs = intent_model(**inputs)
    intent_pred = torch.argmax(intent_outputs.logits, dim=1).item()
    intent_label = id2label_intent[intent_pred]

    # Sentiment Prediction
    sentiment_outputs = sentiment_model(**inputs)
    sentiment_pred = torch.argmax(sentiment_outputs.logits, dim=1).item()
    sentiment_label = id2label_sentiment[sentiment_pred]

    # Display Results
    st.subheader("Predictions:")
    st.write(f"**Intent :**  {intent_label}")
    st.write(f"**Sentiment :**  {sentiment_label}")
