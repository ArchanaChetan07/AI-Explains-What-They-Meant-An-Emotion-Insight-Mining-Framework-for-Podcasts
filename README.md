# AI-Explains-What-They-Meant-An-Emotion-Insight-Mining-Framework-for-Podcasts
AI analyzes podcasts to extract meaning, emotion, and speaker intent.
This project applies advanced text mining techniques to analyze podcast transcripts—extracting core meanings, emotional sentiment, and speaker intent. The pipeline is designed to generate short, AI-narrated summaries that reveal what notable figures *really meant* during conversations. It supports use cases such as quote-based analysis, sentiment visualization, and topic modeling, with an end goal of enabling insightful and accessible video content creation for platforms like YouTube.

---

## Project Objectives

- Extract speaker intent and emotional tone from raw podcast transcripts.
- Perform quote-based analysis to highlight significant statements.
- Visualize sentiment progression using sentiment graphs.
- Apply unsupervised topic modeling to uncover recurring themes and interdisciplinary links.
- Generate short, AI-ready textual summaries for content narration.

---

## Dataset: Whispering-GPT / Lex Fridman Podcast (Hugging Face)

- **Source**: [Hugging Face Dataset](https://huggingface.co/datasets/lex_glue)
- **Size**: Over 140 episodes with timestamped, speaker-labeled transcriptions.
- **Format**: Includes `episode_url`, `title`, `speaker`, `segment_id`, `timestamp_start`, `timestamp_end`, and `transcript`.

---

## Techniques & Tools

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

## Challenges & Roadblocks

- **Whisper model** lacks built-in diarization: built custom segment management logic.
- **Memory limits** when processing large audio files.
- **Topic modeling** required multiple iterations for coherence and interpretability.
- **YouTube scraping** involved handling rate limits and metadata cleanup.

---

## Output

- JSON/CSV files with speaker-labeled quotes and sentiment scores.
- Sentiment timelines for each podcast episode.
- Topic clusters for thematic exploration.
- Dashboard-ready summaries for automated narration or YouTube scripts.
