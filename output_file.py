import os
from datetime import datetime


def save_output(audio, model, transcription):
    file_path = get_output_filepath(audio)
    with open(file_path, 'w') as f:
        f.write(
            f"""
            File: {os.path.basename(audio)}
            Model: {model}
            Transcription: "{transcription}" """)


def save_multiple_output(model, transcription):
    file_path = get_output_filepath()
    with open(file_path, 'w') as f:
        f.write(
            f"""
            Model: {model}
            Transcriptions:
            {transcription}"""
        )


def get_output_filepath(file=None):
    date = datetime.now()
    date_string = date.strftime("%d-%m-%Y")
    time_string = date.strftime("%H-%M-%S")
    if file is None:
        file_path = f"Outputs\\{date_string}\\{time_string}_multiple_output.txt"
    else:
        file_path = f"Outputs\\{date_string}\\{time_string}_{os.path.basename(file)}.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path
