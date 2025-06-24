import os

structure = {
    "notebooks": [
        "01_data_collection.ipynb",
        "02_preprocessing.ipynb",
        "03_topic_modeling.ipynb",
        "04_classification.ipynb",
        "05_flask_integration.ipynb"
    ],
    "data/raw": [
        "lex_fridman_raw.csv"
    ],
    "data/processed": [
        "lex_fridman_cleaned.csv"
    ],
    "visualizations": [],
    "final_report": [
        "AI_Explains_Technical_Notebook.pdf",
        "demo_video.mp4"
    ],
    "app": [
        "app.py",
        "routes.py",
        "utils.py"
    ],
    "app/templates": [
        "index.html"
    ],
    "app/static": [
        "style.css"
    ]
}

# Create folders and placeholder files
for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for file in files:
        with open(os.path.join(folder, file), "w") as f:
            f.write("")

# Top-level files and contents
top_level_files = {
    "README.md": "# AI Explains What They Meant\n\nPodcast NLP Project.",
    "Dockerfile": "",
    "main.py": "",
    "requirements.txt": """\
flask==3.0.2
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
jupyter==1.0.0
nltk==3.8.1
spacy==3.7.4
pyLDAvis==3.4.1
bertopic==0.16.0
tqdm==4.66.2
joblib==1.4.2
sentence-transformers==2.7.0
"""
}

for file, content in top_level_files.items():
    with open(file, "w") as f:
        f.write(content)

print("✅ Project structure and requirements.txt created successfully.")
