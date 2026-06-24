import pytest
import re


class TestTranscriptProcessing:

    def test_speaker_label_extraction(self):
        line = "Speaker 1: I think AI is transforming everything"
        match = re.match(r'^(Speaker \d+):\s*(.*)', line)
        assert match is not None
        assert match.group(1) == "Speaker 1"
        assert "AI" in match.group(2)

    def test_timestamp_removal(self):
        text = "[00:01:23] This is the actual content"
        clean = re.sub(r'\[\d+:\d+:\d+\]\s*', '', text)
        assert clean.strip() == "This is the actual content"

    def test_sentence_splitting(self):
        text = "AI is great. It solves many problems. The future is bright."
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        assert len(sentences) == 3

    def test_empty_transcript_handled(self):
        transcript = ""
        sentences = [s for s in transcript.split('.') if s.strip()]
        assert sentences == []

    def test_keyword_extraction(self):
        text = "machine learning deep learning artificial intelligence neural networks"
        keywords = {"machine learning", "deep learning", "artificial intelligence"}
        found = [k for k in keywords if k in text]
        assert len(found) == 3


class TestEmotionMining:

    def test_emotion_frequency_counts(self):
        emotions = ["joy", "sadness", "joy", "anger", "joy", "sadness"]
        from collections import Counter
        counts = Counter(emotions)
        assert counts["joy"] == 3
        assert counts["sadness"] == 2

    def test_top_emotions_ranked(self):
        from collections import Counter
        emotions = ["joy"]*5 + ["sadness"]*3 + ["anger"]*1
        top = Counter(emotions).most_common(2)
        assert top[0][0] == "joy"
        assert top[1][0] == "sadness"

    def test_segment_length_validation(self):
        segments = ["short", "a medium length segment here", "this is a much longer segment with many more words in it"]
        lengths = [len(s.split()) for s in segments]
        assert lengths[0] < lengths[1] < lengths[2]
