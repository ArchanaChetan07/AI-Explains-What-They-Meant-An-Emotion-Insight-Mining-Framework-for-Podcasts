from flask import Blueprint, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
import joblib
import os

# === 📂 Robust Absolute Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
file_path = os.path.join(BASE_DIR, "notebooks", "data", "processed", "lex_fridman_cleaned.csv")
plot_dir = os.path.join(BASE_DIR, "app", "static", "plots")
os.makedirs(plot_dir, exist_ok=True)

predict_route = Blueprint("predict_route", __name__)

@predict_route.route("/insights")
def show_insights():
    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}", 404

    df = pd.read_csv(file_path)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['duration_minutes'] = df['word_count'] / 150
    df['duration_minutes_rounded'] = df['duration_minutes'].apply(lambda x: round(x / 5) * 5)

    # === 0. Duration Plot ===
    fig, ax = plt.subplots(figsize=(10, 5))
    df['duration_minutes_rounded'].value_counts().sort_index().plot(kind='bar', color='green', ax=ax)
    ax.set_title("Estimated Episode Duration")
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Episode Count")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "duration_plot.png"))
    plt.close(fig)

    # === 1. Top Guests ===
    if 'guest' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df['guest'].value_counts().head(10).plot(kind='bar', color='orange', ax=ax)
        ax.set_title("Top 10 Frequent Guests")
        ax.set_ylabel("Appearances")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "top_guests.png"))
        plt.close(fig)

    # === 2. Word Cloud ===
    all_text = " ".join(df['text'].dropna().tolist())
    wc = WordCloud(width=1000, height=400, background_color='white').generate(all_text)
    wc.to_file(os.path.join(plot_dir, "wordcloud.png"))

    # === 3. Category Trend Plot ===
    if 'published_date' in df.columns and 'category' in df.columns:
        df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
        category_trend = df.groupby([df['published_date'].dt.to_period("M"), 'category']).size().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        category_trend.plot(ax=ax)
        ax.set_title("Category Distribution Over Time")
        ax.set_ylabel("Episode Count")
        ax.set_xlabel("Date")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "category_trend.png"))
        plt.close(fig)

    # === 4. NMF Topics ===
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text'].fillna(""))
    nmf = NMF(n_components=5, random_state=42)
    W_nmf = nmf.fit_transform(tfidf_matrix)
    H_nmf = nmf.components_

    nmf_topics = []
    for topic_idx, topic in enumerate(H_nmf):
        top_words = [tfidf.get_feature_names_out()[i] for i in topic.argsort()[-10:][::-1]]
        nmf_topics.append((f"Topic {topic_idx+1}", top_words))

    nmf_path = os.path.join(plot_dir, "nmf_topics.txt")
    with open(nmf_path, "w", encoding="utf-8") as f:
        for t, words in nmf_topics:
            f.write(f"{t}: {', '.join(words)}\n")

    # === 5. LSA Topics ===
    svd = TruncatedSVD(n_components=5, random_state=42)
    svd.fit(tfidf_matrix)
    lsa_topics = []
    for i, comp in enumerate(svd.components_):
        top_words = [tfidf.get_feature_names_out()[j] for j in np.argsort(comp)[-10:][::-1]]
        lsa_topics.append((f"LSA {i+1}", top_words))

    lsa_path = os.path.join(plot_dir, "lsa_topics.txt")
    with open(lsa_path, "w", encoding="utf-8") as f:
        for t, words in lsa_topics:
            f.write(f"{t}: {', '.join(words)}\n")

    # === 6. LDA Topics ===
    count_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    count_data = count_vectorizer.fit_transform(df['text'].fillna(""))
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(count_data)

    lda_topics = []
    topic_word_scores = []

    for i, topic in enumerate(lda.components_):
        top_words = [count_vectorizer.get_feature_names_out()[j] for j in topic.argsort()[-10:][::-1]]
        lda_topics.append((f"LDA {i+1}", top_words))
        for word in top_words:
            topic_word_scores.append({'Topic': f"LDA {i+1}", 'Word': word})

    lda_path = os.path.join(plot_dir, "lda_topics.txt")
    with open(lda_path, "w", encoding="utf-8") as f:
        for t, words in lda_topics:
            f.write(f"{t}: {', '.join(words)}\n")

    pd.DataFrame(topic_word_scores).to_csv(os.path.join(plot_dir, "topic_word_scores.csv"), index=False)

    # === 7. Classification Report ===
    clf_path = os.path.join(plot_dir, "classification_report.txt")
    try:
        model = joblib.load(os.path.join(BASE_DIR, "final_model", "classification_model.pkl"))
        y_true = df["true_label"] if "true_label" in df.columns else df["category"]
        y_pred = model.predict(tfidf_matrix)
        report = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
        with open(clf_path, "w", encoding="utf-8") as f:
            f.write(report)
    except Exception as e:
        with open(clf_path, "w", encoding="utf-8") as f:
            f.write("⚠️ Classification failed: " + str(e))

    # === Read Generated Files ===
    def read_file(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return "⚠️ File not available."

    nmf_text = read_file(nmf_path)
    lsa_text = read_file(lsa_path)
    lda_text = read_file(lda_path)
    clf_report = read_file(clf_path)

    # === Branding & Monetization ===
    branding = {
        "Curious": "Explore big ideas with wonder.",
        "Conscious": "Ethics, meaning, and self-awareness.",
        "Cutting-Edge": "Stay on the frontier of science & AI."
    }

    monetization = {
        "Sponsorships": ["AI tools", "Edtech platforms", "Wellness apps"],
        "Premium Content": ["Uncut interviews", "Live AMAs"],
        "Cross-Platform": ["YouTube Shorts", "Newsletter", "LinkedIn growth"]
    }

    return render_template("insights.html",
        branding=branding,
        monetization=monetization,
        nmf_text=nmf_text,
        lsa_text=lsa_text,
        lda_text=lda_text,
        clf_report=clf_report
    )

