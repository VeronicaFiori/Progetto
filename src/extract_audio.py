# src/extract_audio.py
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path

def extract_audio_from_mp4(mp4_path, output_wav_path):
    video = VideoFileClip(mp4_path)
    audio = video.audio
    audio.write_audiofile(output_wav_path)

def process_folder(mp4_folder, output_folder):
    if not os.path.exists(mp4_folder):
        print(f"❌ La cartella non esiste: {mp4_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(mp4_folder):
        if file.endswith(".mp4"):
            mp4_path = os.path.join(mp4_folder, file)
            wav_path = os.path.join(output_folder, file.replace(".mp4", ".wav"))
            extract_audio_from_mp4(mp4_path, wav_path)
            print(f"✓ Audio estratto: {wav_path}")

if __name__ == "__main__":
    # Trova il percorso Desktop in modo compatibile con o senza OneDrive
    possible_desktops = [
        Path.home() / "OneDrive" / "Desktop" / "dati",
        Path.home() / "Desktop" / "dati"
    ]

    for path in possible_desktops:
        if path.exists():
            base_data = path
            break
    else:
        print("❌ Cartella 'dati' non trovata né su OneDrive né su Desktop.")
        exit(1)

    mp4_folder = base_data / "mp4"
    wav_folder = base_data / "wav"

    if not mp4_folder.exists():
        print(f"❌ La cartella non esiste: {mp4_folder}")
        exit(1)

    process_folder(str(mp4_folder), str(wav_folder))

