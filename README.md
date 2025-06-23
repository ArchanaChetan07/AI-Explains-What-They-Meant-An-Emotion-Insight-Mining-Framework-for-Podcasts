
# 🎙️ AI Explains What They Meant: A Conversational AI Framework for Podcast Intelligence

**AI Explains What They Meant** is a professional-grade, full-stack NLP pipeline and LangChain-based conversational AI system designed to extract, interpret, and interact with podcast transcript content—specifically tailored to the **Lex Fridman Podcast** dataset. The project is built to enable quote-level topic analysis, guest exploration, and interactive explanations via a multimodal UI powered by GPT-4.

---

## Project Highlights

-  **Audio to Text**: Extracts transcripts from YouTube podcasts using Whisper ASR.
-  **Advanced NLP Pipeline**: Cleans, structures, and analyzes speaker-labeled text.
-  **Unsupervised Topic Modeling**: NMF, LSA, and LDA applied to ungrouped podcast segments.
-  **Quote Classification**: Predicts the topic category using TF-IDF + ML model.
-  **LangChain Agent**: GPT-4-powered assistant integrates knowledge with tools.
-  **Flask Dashboard**: Interactive data visualization via HTML templates.
-  **Streamlit Bot**: Frontend chatbot that leverages LangChain agent tools.
-  **Dockerized**: Full stack deployment-ready via Docker.

---

## Project Structure

```
AI_Explains_What_They_Meant/
├── app/
│   ├── app.py                  # Flask entry point
│   ├── routes.py               # Insight route logic
│   ├── utils.py                # Model loader and predict function
│   ├── chat_logger.py          # Chat logging to JSON
│   ├── langchain_chat.py       # GPT-4-based prompt chain
│   ├── templates/
│   │   ├── index.html
│   │   └── insights.html
│   └── tools/
│       ├── predict_topic.py
│       ├── guest_search.py
│       ├── wiki_tool.py
│       ├── langchain_service.py
│       └── __init__.py
├── notebooks/                  # Full NLP pipeline
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_topic_modeling.ipynb
│   ├── 04_classification.ipynb
│   └── 05_flask_integration.ipynb
├── streamlit/
│   └── app_streamlit.py        # Conversational UI
├── final_model/                # Saved model & vectorizer
├── final_report/               # PDF + Video deliverables
├── static/                     # Plot assets
├── visualizations/             # External media
├── Dockerfile                  # Containerization
├── requirements.txt            # Environment dependencies
├── .env                        # OpenAI API key config
├── .gitignore                  # Version control hygiene
├── README.md
└── scripts/
    └── create_project_structure.py
```

---

## Quickstart Guide

### Setup Environment

```bash
git clone https://github.com/your-org/AI-Explains-What-They-Meant.git
cd AI-Explains-What-They-Meant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your `.env` file with:
```bash
OPENAI_API_KEY=your-openai-api-key
```

---

### Run the Flask App (Insights Dashboard)

```bash
cd app
python app.py
```

Visit: [http://127.0.0.1:5000/insights](http://127.0.0.1:5000/insights)

---

###  Run the Streamlit App (LangChain Chatbot)

```bash
streamlit run streamlit/app_streamlit.py
```

---

## LangChain Agent Tooling

| Tool Name        | Description |
|------------------|-------------|
| `PredictTopic`   | Classify any quote into one of 5 podcast categories |
| `WikiSearch`     | Lookup summaries from Wikipedia |
| `ListAllGuests`  | Extracts and displays known guests from dataset |

---

##  NLP Stack

- **ASR**: OpenAI Whisper
- **Modeling**: TF-IDF + LogisticRegression
- **Topic Models**: NMF, LSA, LDA
- **Embeddings**: (Optional) `sentence-transformers` support
- **LangChain**: Conversational Agent + Tools + GPT-4
---

##  Docker Deployment

```bash
docker build -t ai-explains-bot .
docker run -p 5000:5000 ai-explains-bot
```
---
## Developed By

Archana Suresh Patil  
University of San Diego | Applied Data Science 509  
GitHub: [ArchanaChetan07](https://github.com/ArchanaChetan07)

---

## Academic Context: ADS-509 Project Submission

This project was submitted for **ADS-509 – Applied Text Mining** at the University of San Diego.

**Team Number**: Group 6  
**Team Leader**: Archana Suresh Patil  
**Project Title**: *AI Explains What They Meant: An Emotion & Insight Mining Framework for Podcasts*  
**Dataset**: Lex Fridman Podcast Dataset (via Hugging Face)  
[GitHub Repository](https://github.com/ArchanaChetan07/AI-Explains-What-They-Meant.git)

### 🧾 Objectives (from official project update)
This project applies advanced text mining techniques to analyze podcast transcripts, extracting core meanings, emotional sentiment, and speaker intent. The objective is to build a pipeline that generates short AI-narrated explanations of what notable figures really meant, using quote-based analysis, sentiment graphs, and topic modeling.

The final output supports video content creation for platforms like YouTube and explores how natural language understanding can enhance public access to complex conversations.

###  Dataset Overview
- **Source**: Hugging Face – Whispering-GPT/Lex Fridman Podcast Dataset
- **Volume**: 140+ episodes transcribed via Whisper
- **Variables**: `episode_title`, `speaker`, `text`, `timestamp_start/end`, `segment_id`
- **Additional**: Custom YouTube scraper for latest 10 episodes via `yt-dlp` + Whisper Large

### 🛠 Challenges Encountered
- **Heavy compute demand** from Whisper transcription
- **Speaker alignment** without diarization required custom parsing
- **Topic coherence tuning** took multiple iterations
- **Scraping stability** required retry logic due to YouTube rate limits

---

---

#  AI-Explains-What-They-Meant: An Emotion & Insight Mining Framework for Podcasts

AI analyzes podcasts to extract meaning, emotion, and speaker intent.

This project applies advanced text mining techniques to analyze podcast transcripts—extracting core meanings, emotional sentiment, and speaker intent. The pipeline is designed to generate short, AI-narrated summaries that reveal what notable figures *really meant* during conversations. It supports use cases such as quote-based analysis, sentiment visualization, and topic modeling, with an end goal of enabling insightful and accessible video content creation for platforms like YouTube.

---

##  Project Objectives

- Extract speaker intent and emotional tone from raw podcast transcripts.
- Perform quote-based analysis to highlight significant statements.
- Visualize sentiment progression using sentiment graphs.
- Apply unsupervised topic modeling to uncover recurring themes and interdisciplinary links.
- Generate short, AI-ready textual summaries for content narration.

---

## Lex Fridman Podcast (Hugging Face)

- **Source**: [Hugging Face Dataset](https://huggingface.co/datasets/lex_glue)
- **Size**: Over 140 episodes with timestamped, speaker-labeled transcriptions.
- **Format**: Includes `episode_url`, `title`, `speaker`, `segment_id`, `timestamp_start`, `timestamp_end`, and `transcript`.

---

##  Techniques & Tools

| Category               | Tool / Method                                      |
|------------------------|----------------------------------------------------|
| **Transcription**      | OpenAI Whisper Large                               |
| **Preprocessing**      | Text normalization, custom speaker diarization     |
| **Sentiment Analysis** | Lexicon-based scoring, Vader, TextBlob             |
| **Topic Modeling**     | LDA, BERTopic, TF-IDF                              |
| **Summarization**      | spaCy, BART (optional integration)                 |
| **Visualization**      | Matplotlib, Seaborn, WordCloud                     |
| **Web Scraping**       | yt-dlp, BeautifulSoup                              |

---

##  Challenges & Roadblocks

- **Whisper model** lacks built-in diarization: built custom segment management logic.
- **Memory limits** when processing large audio files.
- **Topic modeling** required multiple iterations for coherence and interpretability.
- **YouTube scraping** involved handling rate limits and metadata cleanup.

---

##  Output Artifacts

- JSON/CSV files with speaker-labeled quotes and sentiment scores.
- Sentiment timelines for each podcast episode.
- Topic clusters for thematic exploration.
- Dashboard-ready summaries for automated narration or YouTube scripts.

