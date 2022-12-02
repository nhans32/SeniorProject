from __future__ import unicode_literals
import librosa
import numpy as np
from threading import Thread
import joblib
import tensorflow as tf
import yt_dlp
import os
from flask import Flask
from flask import abort
from flask_cors import CORS 
from flask import request
import tempo_algorithms
from yt_dlp import utils as yt_dlp_utils

app = Flask(__name__)
cors = CORS(app)

ydl_opts = {
    'format': 'wav/bestaudio/best',
    'format_sort': ['+size', '+br', '+res', '+fps'],
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
    'paths': {'temp': "Application/backend/temp",
                'home': "Application/backend/temp"},
    'restrictfilenames': True
}

genre_classifier = joblib.load("Models/svc_genre_classifier.joblib")
genre_scaler = joblib.load("Util/genre_scaler.joblib")
genre_convertor = joblib.load("Util/genre_convertor.joblib")

mood_classifier = tf.keras.models.load_model("Models/nn_mood_classifier")
mood_scaler = joblib.load("Util/mood_scaler.joblib")
mood_convertor = joblib.load("Util/mood_convertor.joblib")

def chroma_stft(audio_data, sr, feature_dict):
    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    feature_dict["chroma_stft_mean"] = np.mean(chroma_stft)
    feature_dict["chroma_stft_var"] = np.var(chroma_stft)
    print("Chroma completed")

def rms(audio_data, sr, feature_dict):
    rms = librosa.feature.rms(y=audio_data)
    feature_dict["rms_mean"] = np.mean(rms)
    feature_dict["rms_var"] = np.var(rms)
    print("Rms Completed")

def spec_centroid(audio_data, sr, feature_dict):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    feature_dict["spec_centroid_mean"] = np.mean(spectral_centroid)
    feature_dict["spec_centroid_var"] = np.var(spectral_centroid)
    print("Spec centroid completed")

def spec_bandwidth(audio_data, sr, feature_dict):
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
    feature_dict["spec_bandwidth_mean"] = np.mean(spectral_bandwidth)
    feature_dict["spec_bandwidth_var"] = np.var(spectral_bandwidth)
    print("Spec bandwidth completed")

def spec_rolloff(audio_data, sr, feature_dict):
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
    feature_dict["spec_rolloff_mean"] = np.mean(spectral_rolloff)
    feature_dict["spec_rolloff_var"] = np.var(spectral_rolloff)
    print("Spec rolloff completed")

def zero_cross(audio_data, sr, feature_dict):
    zero_cross = librosa.feature.zero_crossing_rate(y=audio_data)
    feature_dict["zero_cross_mean"] = np.mean(zero_cross)
    feature_dict["zero_cross_var"] = np.var(zero_cross)
    print("Zero cross completed")

def harm_perc(audio_data, sr, feature_dict):
    harmony, perceptrual = librosa.effects.hpss(y=audio_data)
    feature_dict["harmony_mean"] = np.mean(harmony)
    feature_dict["harmony_var"] = np.var(harmony)
    feature_dict["perc_mean"] = np.mean(perceptrual)
    feature_dict["perc_var"] = np.var(perceptrual)
    print("Harmony and Perc completed")

def tempo(audio_data, sr, feature_dict):
    tempo = librosa.beat.tempo(y=audio_data, sr=sr)
    feature_dict["tempo"] = np.mean(tempo)
    print("Tempo completed")

def mfcc(audio_data, sr, feature_dict):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr)
    for idx, mfcc in enumerate(mfccs):
        feature_dict[f"mfcc{idx+1}_mean"] = np.mean(mfcc)
        feature_dict[f"mfcc{idx+1}_var"] = np.var(mfcc)
    print("MFCC completed")

def extract_features(file_path):
    print(f"Processing file: {file_path}")

    audio_data, sr = librosa.load(file_path)
    audio_data = librosa.effects.trim(audio_data)[0] # trim the audio data

    feature_dict = {}

    threads = []

    print("Process 1: chroma_stft")
    chroma_stft_process = Thread(target=chroma_stft, args=(audio_data, sr, feature_dict))
    threads.append(chroma_stft_process)
    chroma_stft_process.start()

    print("Process 2: rms")
    rms_process = Thread(target=rms, args=(audio_data, sr, feature_dict))
    threads.append(rms_process)
    rms_process.start()

    print("Process 3: spec_centroid")
    spec_centroid_process = Thread(target=spec_centroid, args=(audio_data, sr, feature_dict))
    threads.append(spec_centroid_process)
    spec_centroid_process.start()

    print("Process 4: spec_bandwidth")
    spec_bandwidth_process = Thread(target=spec_bandwidth, args=(audio_data, sr, feature_dict))
    threads.append(spec_bandwidth_process)
    spec_bandwidth_process.start()

    print("Process 5: spec_rolloff")
    spec_rolloff_process = Thread(target=spec_rolloff, args=(audio_data, sr, feature_dict))
    threads.append(spec_rolloff_process)
    spec_rolloff_process.start()

    print("Process 6: zero_cross")
    zero_cross_process = Thread(target=zero_cross, args=(audio_data, sr, feature_dict))
    threads.append(zero_cross_process)
    zero_cross_process.start()

    print("Process 7: harm_perc")
    harm_perc_process = Thread(target=harm_perc, args=(audio_data, sr, feature_dict))
    threads.append(harm_perc_process)
    harm_perc_process.start()

    print("Process 8: tempo")
    tempo_process = Thread(target=tempo, args=(audio_data, sr, feature_dict))  
    threads.append(tempo_process)
    tempo_process.start()

    print("Process 9: mfcc")
    mfcc_process = Thread(target=mfcc, args=(audio_data, sr, feature_dict))
    threads.append(mfcc_process)
    mfcc_process.start()
    
    for proc in threads:
        proc.join()

    print(f"Completed: {file_path}")
    return audio_data, sr, feature_dict

def construct_feat_arr(feature_dict):
    feat_arr = []
    feat_arr.append(feature_dict['chroma_stft_mean'])
    feat_arr.append(feature_dict['chroma_stft_var'])
    feat_arr.append(feature_dict['rms_mean'])
    feat_arr.append(feature_dict['rms_var'])
    feat_arr.append(feature_dict['spec_centroid_mean'])
    feat_arr.append(feature_dict['spec_centroid_var'])
    feat_arr.append(feature_dict['spec_bandwidth_mean'])
    feat_arr.append(feature_dict['spec_bandwidth_var'])
    feat_arr.append(feature_dict['spec_rolloff_mean'])
    feat_arr.append(feature_dict['spec_rolloff_var'])
    feat_arr.append(feature_dict['zero_cross_mean'])
    feat_arr.append(feature_dict['zero_cross_var'])
    feat_arr.append(feature_dict['harmony_mean'])
    feat_arr.append(feature_dict['harmony_var'])
    feat_arr.append(feature_dict['perc_mean'])
    feat_arr.append(feature_dict['perc_var'])
    feat_arr.append(feature_dict['tempo'])
    
    for i in range(1, 21):
        feat_arr.append(feature_dict[f'mfcc{i}_mean'])
        feat_arr.append(feature_dict[f'mfcc{i}_var'])

    return np.array(feat_arr)

def classify(feat_arr):
    feat_arr_genre = genre_scaler.transform([feat_arr])
    feat_arr_mood = mood_scaler.transform([feat_arr])

    genre_probs = genre_classifier.predict_proba(feat_arr_genre)
    mood_probs = mood_classifier.predict(feat_arr_mood)

    return genre_probs, mood_probs

@app.route("/classify", methods=["POST"])
def classify_route():
    if request.is_json: # its a youtube link         
        yt_link = request.json['link']
        tempo_algo = request.json['tempoAlgo']

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(yt_link, download=False)
            info_dict = ydl.sanitize_info(info) # TODO: CHECK IF ITS A PLAYLIST
            if info_dict['duration'] > 600:
                abort(413)
            error_code = ydl.download([yt_link])

        vid_title = yt_dlp_utils.sanitize_filename(info_dict['title'], True)
        filename = vid_title + '-' + '[' + info_dict['id'] + ']' + ".wav"

        fp = "Application/backend/temp" + os.sep + filename

    else: # its a file
        file = request.files['file']
        tempo_algo = request.form['tempoAlgo']

        filename = file.filename.replace("/", "_").replace("\\", "_")
        # check if wav file
        if filename.endswith(".wav"):
            fp = "Application/backend/temp" + os.sep + filename
            file.save(fp)
        else:
            abort(415)

    audio_data, sr, feature_dict = extract_features(fp)
    feat_arr = construct_feat_arr(feature_dict)

    genre_probs, mood_probs = classify(feat_arr)

    genre_eng_pred = genre_convertor.inverse_transform(np.argmax(genre_probs, axis=1))[0]
    mood_eng_pred = mood_convertor.inverse_transform(np.argmax(mood_probs, axis=1))[0]

    genre_conf_pred = genre_probs[0][np.argmax(genre_probs, axis=1)[0]]
    mood_conf_pred = mood_probs[0][np.argmax(mood_probs, axis=1)[0]]

    if tempo_algo == "lfa":
        print("Low Pass Filter")
        pred_tempo = round(float(tempo_algorithms.lpf_algorithm((audio_data, sr))), 2)
    elif tempo_algo == "sea":
        print("Sound Energy")
        pred_tempo = round(float(tempo_algorithms.sound_energy_algorithm((audio_data, sr))), 2)
    elif tempo_algo == "noa":
        print("Note Onset")
        pred_tempo = round(float(tempo_algorithms.note_onset_algorithm((audio_data, sr))), 2)
    else:
        pred_tempo = "NULL"

    os.remove(fp)

    return {"song_title": filename,
            "tempo": pred_tempo,
            "genre": 
                {
                "prediction": genre_eng_pred,
                "confidence": float(genre_conf_pred),
                "probabilities": [{"genre": genre_convertor.classes_[i], "probability": genre_probs[0].tolist()[i]} for i in range(len(genre_probs[0].tolist()))]
                }, 
            "mood": 
                {
                "prediction": mood_eng_pred,
                "confidence": float(mood_conf_pred),
                "probabilities": [{"mood": mood_convertor.classes_[i], "probability": mood_probs[0].tolist()[i]} for i in range(len(mood_probs[0].tolist()))]
                }
            }

if __name__ == "__main__":
    app.run(debug=True, port=8000)