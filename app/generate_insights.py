import matplotlib
matplotlib.use('Agg')  # ✅ Prevent GUI errors in Flask

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
import joblib
import os

# ========= 📁 ROOT PATH =========
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "notebooks", "data", "processed", "lex_fridman_cleaned.csv")
PLOT_DIR = os.path.join(BASE_DIR, "app", "static", "plots")
MODEL_PATH = os.path.join(BASE_DIR, "final_model", "classification_model.pkl")

os.makedirs(PLOT_DIR, exist_ok=True)

# ========= 📊 Load Data =========
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ CSV not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# ========= 🕒 Duration Plot =========
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['duration_minutes'] = df['word_count'] / 150
df['duration_minutes_rounded'] = df['duration_minutes'].apply(lambda x: round(x / 5) * 5)

plt.figure(figsize=(10, 5))
df['duration_minutes_rounded'].value_counts().sort_index().plot(kind='bar', color='green')
plt.title("Estimated Episode Duration")
plt.xlabel("Minutes")
plt.ylabel("Episode Count")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "duration_plot.png"))
plt.close()

# ========= 👥 Top 10 Guests =========
if 'guest' in df.columns:
    guest_counts = df['guest'].value_counts().head(10)
    plt.figure(figsize=(10, 5))
    guest_counts.plot(kind='bar', color='orange')
    plt.title("Top 10 Frequent Guests")
    plt.ylabel("Appearances")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "top_guests.png"))
    plt.close()

# ========= ☁️ Word Cloud =========
all_text = " ".join(df['text'].dropna().tolist())
wc = WordCloud(width=1000, height=400, background_color='white').generate(all_text)
wc.to_file(os.path.join(PLOT_DIR, "wordcloud.png"))

# ========= 📆 Category Over Time =========
if 'published_date' in df.columns and 'category' in df.columns:
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    category_trend = df.groupby([df['published_date'].dt.to_period("M"), 'category']).size().unstack().fillna(0)
    category_trend.plot(figsize=(12, 6))
    plt.title("Category Distribution Over Time")
    plt.ylabel("Episode Count")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "category_trend.png"))
    plt.close()

# ========= 🧠 NMF Topics =========
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'].fillna(""))

nmf = NMF(n_components=5, random_state=42)
W_nmf = nmf.fit_transform(tfidf_matrix)
H_nmf = nmf.components_

nmf_topics = []
for topic_idx, topic in enumerate(H_nmf):
    top_words = [tfidf.get_feature_names_out()[i] for i in topic.argsort()[-10:][::-1]]
    nmf_topics.append((f"Topic {topic_idx+1}", top_words))

with open(os.path.join(PLOT_DIR, "nmf_topics.txt"), "w", encoding="utf-8") as f:
    for t, words in nmf_topics:
        f.write(f"{t}: {', '.join(words)}\n")

# ========= 📚 LSA Topics =========
svd = TruncatedSVD(n_components=5, random_state=42)
svd.fit(tfidf_matrix)

lsa_topics = []
for i, comp in enumerate(svd.components_):
    top_words = [tfidf.get_feature_names_out()[j] for j in np.argsort(comp)[-10:][::-1]]
    lsa_topics.append((f"LSA {i+1}", top_words))

with open(os.path.join(PLOT_DIR, "lsa_topics.txt"), "w", encoding="utf-8") as f:
    for t, words in lsa_topics:
        f.write(f"{t}: {', '.join(words)}\n")

# ========= 🧵 LDA Topics =========
count_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
count_data = count_vectorizer.fit_transform(df['text'].fillna(""))

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(count_data)

lda_topics = []
for i, topic in enumerate(lda.components_):
    top_words = [count_vectorizer.get_feature_names_out()[j] for j in topic.argsort()[-10:][::-1]]
    lda_topics.append((f"LDA {i+1}", top_words))

with open(os.path.join(PLOT_DIR, "lda_topics.txt"), "w", encoding="utf-8") as f:
    for t, words in lda_topics:
        f.write(f"{t}: {', '.join(words)}\n")

# ========= 📈 Topic Word Scores =========
topic_word_df = pd.DataFrame(H_nmf.T, index=tfidf.get_feature_names_out())
topic_word_df.to_csv(os.path.join(PLOT_DIR, "topic_word_scores.csv"))

# ========= 🧪 Classification Report =========
try:
    model = joblib.load(MODEL_PATH)
    y_true = df["true_label"] if "true_label" in df.columns else df["category"]
    y_pred = model.predict(tfidf_matrix)
    report = classification_report(y_true, y_pred, output_dict=False, zero_division=0)
    with open(os.path.join(PLOT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
except Exception as e:
    print("⚠️ Classification model failed:", e)

print("✅ All plots and files have been saved to:", PLOT_DIR)
