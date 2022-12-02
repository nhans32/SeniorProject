import sys
sys.path.append('/path/to/ffmpeg')
import os
from pydub import AudioSegment
import shutil

def convert_audio(file_path):
    print(f"Processing file: {file_path}")

    if file_path.endswith(".mp3"):
        sound = AudioSegment.from_mp3(file_path)
        sound.export(file_path.replace(".mp3", ".wav"), format="wav")
        os.remove(file_path)
    elif file_path.endswith(".wav"):
        print(f"File {file_path} already in wav format..")
    else:
        print(f"File {file_path} is not supported")


if __name__ == "__main__":
    tempo_path = "RawTempoData/tempo"
    audio_path = "RawTempoData/audio"

    out_path = "TempoData"
    
    for file in os.listdir(audio_path):
        convert_audio(audio_path + "/" + file)

    for file in os.listdir(tempo_path):
        with open(tempo_path + "/" + file, "r") as fp:
            bpm = fp.readline().strip()
            new_path = out_path + "/" + bpm
            audio_file = file.replace(".bpm", ".wav")

            if not os.path.exists(new_path):
                os.makedirs(new_path)
            shutil.move(audio_path + "/" + audio_file, new_path + "/" + audio_file)