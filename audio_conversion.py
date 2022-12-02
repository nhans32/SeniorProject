from pydub import AudioSegment
import concurrent.futures
import os

# NOTE: RUN THIS BEFORE RUNNING AUDIO_FEATURE_EXTRACT.PY

DATA_PATH = "Data" + os.sep

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


def conversion_multiprocess(files_labels):
    for files_label in files_labels:
        print(f"Processing class {files_label[1]}...")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(convert_audio, files_label[0])

def create_files_labels(data_dir, allowed_taxonomies, allowed_classes):
    files_labels = []

    for folder in os.listdir(data_dir):
        if folder in allowed_taxonomies:
            for subfolder in os.listdir(data_dir + folder):
                if subfolder in allowed_classes:
                    if os.path.isdir(data_dir + folder + os.sep + subfolder): # check if subfolder is a directory
                        class_arr = ([], folder + os.sep + subfolder)
                        for file in os.listdir(data_dir + folder + os.sep + subfolder):
                            class_arr[0].append(data_dir + folder + os.sep + subfolder + os.sep + file)

                        files_labels.append(class_arr)
    return files_labels

if __name__ == "__main__":
    allowed_taxonomies = {"genre"}
    allowed_classes = {"classical"}

    files_labels = create_files_labels(DATA_PATH, allowed_taxonomies=allowed_taxonomies, allowed_classes=allowed_classes) # list of files and their corresponding labels stored in tuples
    conversion_multiprocess(files_labels)


