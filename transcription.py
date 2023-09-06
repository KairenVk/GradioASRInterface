import os
import output_file
import gradio as gr
from transformers import pipeline
from settings import models_dir


def transcribe(audio, model):
    gr.Info("Downloading selected model...")
    pipe = pipeline(model=model)
    pipe.save_pretrained(models_dir)
    gr.Info("Transcribing file...")
    result = pipe(audio, chunk_length_s=10)['text']
    output_file.save_output(audio, model, result)
    return result


def transcribe_multiple(audio, model):
    transcriptions = []
    files = []
    result = ""
    for file in audio:
        files.append(file.name)
    pipe = pipeline(model=model)
    pipe.save_pretrained(models_dir)
    for file in files:
        transcriptions.append(pipe(file, chunk_length_s=10)['text'])
    for x in range(len(transcriptions)):
        result += (os.path.basename(files[x]) + ": " + "\""+transcriptions[x]+"\"" + "\n")
    output_file.save_multiple_output(model, result)
    return result
