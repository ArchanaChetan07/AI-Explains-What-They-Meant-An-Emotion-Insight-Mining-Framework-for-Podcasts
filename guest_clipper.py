import yt_dlp
import os
import whisper
import warnings
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# 📁 Configuration
youtube_urls = [
    "https://www.youtube.com/watch?v=_1f-o0nqpEI"
]
download_dir = r"C:\Users\archa\Desktop\Clip youtube"
ffmpeg_path = r"C:\Users\archa\Downloads\ffmpeg-2025-05-15-git-12b853530a-essentials_build\ffmpeg-2025-05-15-git-12b853530a-essentials_build\bin"

# 🧠 Set FFmpeg in system PATH
os.environ["PATH"] += os.pathsep + ffmpeg_path

# ✅ Download MP3 from YouTube and return the downloaded MP3 filename
def download_audio_only(url, path):
    if not os.path.exists(path):
        os.makedirs(path)

    audio_opts = {
        'outtmpl': os.path.join(path, '%(title).50s_audio.%(ext)s'),
        'format': 'bestaudio/best',
        'ffmpeg_location': ffmpeg_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    print(f"📥 Downloading: {url}")
    with yt_dlp.YoutubeDL(audio_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        title = info_dict.get('title', 'audio')
        output_filename = f"{title[:50]}_audio.mp3"
        return os.path.join(path, output_filename)

# 🎙️ Transcribe audio using Whisper
def transcribe_audio(file_path):
    model = whisper.load_model("base")

    if torch.cuda.is_available():
        print("✅ Using GPU")
    else:
        print("⚠️ Using CPU")

    result = model.transcribe(file_path, fp16=False, language="en", task="transcribe")

    transcript_file = file_path.replace(".mp3", "_transcription.txt")
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print(f"📝 Transcript saved to: {transcript_file}")

# 🚀 Main execution
if __name__ == "__main__":
    for url in youtube_urls:
        mp3_file_path = download_audio_only(url, download_dir)
        print(f"\n🎧 Transcribing: {os.path.basename(mp3_file_path)}")
        transcribe_audio(mp3_file_path)

    print("\n✅ All done: only new audio transcribed.")
