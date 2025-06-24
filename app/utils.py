import os
import joblib
import numpy as np

# === Load Model and Vectorizer ===
def load_model_and_vectorizer():
    model_path = "../final_model/classification_model.pkl"
    vectorizer_path = "../final_model/vectorizer.pkl"

    assert os.path.exists(model_path), "❌ classification_model.pkl not found"
    assert os.path.exists(vectorizer_path), "❌ vectorizer.pkl not found"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ Model and vectorizer loaded successfully!")
    return model, vectorizer

# === Topic Labels ===
topic_labels = {
    0: "AI Ethics & Alignment",
    1: "Science & Physics",
    2: "Programming & Tech",
    3: "Neuroscience & Consciousness",
    4: "Biology & Medicine"
}

# === Predict Function ===
def predict_topic(text, model, vectorizer, top_n_words=5):
    vec = vectorizer.transform([text])
    topic = model.predict(vec)[0]
    dense_vec = vec.todense().tolist()[0]
    feature_names = vectorizer.get_feature_names_out()

    top_indices = np.argsort(dense_vec)[::-1][:top_n_words]
    top_words = [(feature_names[i], dense_vec[i]) for i in top_indices if dense_vec[i] > 0]

    label = topic_labels.get(topic, "Unknown")

    return {
        "predicted_topic": int(topic),
        "predicted_label": label,
        "top_words": top_words
    }
