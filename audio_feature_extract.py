import concurrent.futures
from audio_conversion import create_files_labels
import os
import librosa
import pandas as pd
import numpy as np

DATA_PATH = "Data" + os.sep
AUDIO_FEATURES = ["chroma_stft_mean",
"chroma_stft_var",
"rms_mean",
"rms_var",
"spectral_centroid_mean",
"spectral_centroid_var",
"spectral_bandwidth_mean",
"spectral_bandwidth_var",
"rolloff_mean",
"rolloff_var",
"zero_crossing_rate_mean",
"zero_crossing_rate_var",
"harmony_mean",
"harmony_var",
"perceptr_mean",
"perceptr_var",
"tempo",
"mfcc1_mean",
"mfcc1_var",
"mfcc2_mean",
"mfcc2_var",
"mfcc3_mean",
"mfcc3_var",
"mfcc4_mean",
"mfcc4_var",
"mfcc5_mean",
"mfcc5_var",
"mfcc6_mean",
"mfcc6_var",
"mfcc7_mean",
"mfcc7_var",
"mfcc8_mean",
"mfcc8_var",
"mfcc9_mean",
"mfcc9_var",
"mfcc10_mean",
"mfcc10_var",
"mfcc11_mean",
"mfcc11_var",
"mfcc12_mean",
"mfcc12_var",
"mfcc13_mean",
"mfcc13_var",
"mfcc14_mean",
"mfcc14_var",
"mfcc15_mean",
"mfcc15_var",
"mfcc16_mean",
"mfcc16_var",
"mfcc17_mean",
"mfcc17_var",
"mfcc18_mean",
"mfcc18_var",
"mfcc19_mean",
"mfcc19_var",
"mfcc20_mean",
"mfcc20_var"]

def extract_features_from_class(file_path):
    print(f"Processing file: {file_path}")
    feature_arr = []

    audio_data, sr = librosa.load(file_path)
    audio_data = librosa.effects.trim(audio_data)[0] # trim the audio data

    feature_arr.append(file_path)

    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    feature_arr.append(np.mean(chroma_stft))
    feature_arr.append(np.var(chroma_stft))

    rms = librosa.feature.rms(y=audio_data)
    feature_arr.append(np.mean(rms))
    feature_arr.append(np.var(rms))

    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    feature_arr.append(np.mean(spectral_centroid))
    feature_arr.append(np.var(spectral_centroid))

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
    feature_arr.append(np.mean(spectral_bandwidth))
    feature_arr.append(np.var(spectral_bandwidth))

    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
    feature_arr.append(np.mean(spectral_rolloff))
    feature_arr.append(np.var(spectral_rolloff))

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)
    feature_arr.append(np.mean(zero_crossing_rate))
    feature_arr.append(np.var(zero_crossing_rate))

    harmony, perceptrual = librosa.effects.hpss(y=audio_data)
    feature_arr.append(np.mean(harmony))
    feature_arr.append(np.var(harmony))
    feature_arr.append(np.mean(perceptrual))
    feature_arr.append(np.var(perceptrual))

    tempo = librosa.beat.tempo(y=audio_data, sr=sr)
    feature_arr.append(np.mean(tempo))

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr)
    for mfcc in mfccs:
        feature_arr.append(np.mean(mfcc))
        feature_arr.append(np.var(mfcc))

    print(f"Completed: {file_path}")
    return feature_arr

def features_from_class(files, label):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        features = executor.map(extract_features_from_class, files)

    features_arr = [feat + [label] for feat in features]
    feature_cols = ["filepath"] + AUDIO_FEATURES + ["label"]

    return pd.DataFrame(features_arr, columns=feature_cols)


def process_classes(files_labels):
    for files_label in files_labels:
        print(f"Processing class {files_label[1]}...")
        df = features_from_class(files_label[0], files_label[1].split(os.sep)[-1])
        df.to_csv(f"{DATA_PATH + files_label[1]}.csv", index=False)

def combine_csvs(DATA_PATH):
    for dir_name in os.listdir(DATA_PATH):
        csv_files = [f for f in os.listdir(DATA_PATH + dir_name) if f.endswith(".csv") and "_features.csv" not in f]
        if len(csv_files) != 0:
            df = pd.concat([pd.read_csv(DATA_PATH + dir_name + os.sep + f) for f in csv_files])
            df.to_csv(DATA_PATH + dir_name + os.sep + dir_name + "_" + "features" + ".csv", index=False)

if __name__ == "__main__":
    allowed_taxonomies = {"genre"}
    allowed_classes = {"classical"}

    # change to multithreading

    files_labels = create_files_labels(DATA_PATH, allowed_taxonomies=allowed_taxonomies, allowed_classes=allowed_classes) # list of files and their corresponding labels stored in tuples
    process_classes(files_labels)
    combine_csvs(DATA_PATH)