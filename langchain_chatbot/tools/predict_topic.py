from langchain.tools import tool
import joblib
import os
import numpy as np

# Load model + vectorizer
model = joblib.load("final_model/classification_model.pkl")
vectorizer = joblib.load("final_model/vectorizer.pkl")

# Topic label map
topic_labels = {
    0: "AI Ethics & Alignment",
    1: "Science & Physics",
    2: "Programming & Tech",
    3: "Neuroscience & Consciousness",
    4: "Biology & Medicine"
}

@tool
def predict_topic(text: str) -> str:
    """Predicts the topic of a podcast segment."""
    vec = vectorizer.transform([text])
    topic = model.predict(vec)[0]
    label = topic_labels.get(topic, "Unknown")
    return f"The segment is about: {label}"
