import os

from huggingface_hub import HfApi, ModelFilter
import torch
from transformers import pipeline
import settings

models_dir = settings.models_dir


def getModels():
    api = HfApi()
    models = (api.list_models(sort="downloads", direction=-1, limit=10,
                              filter=ModelFilter(
                                  task="automatic-speech-recognition",
                                  language="pl"
                              )
                              ))
    return models


def transcribe(audio, model):
    device = torch.device(0)
    pipe = pipeline(model=model)
    pipe.save_pretrained(models_dir)
    result = pipe(audio, chunk_length_s=10)['text']
    return result


def transcribe_multiple(audio, model):
    transcriptions = []
    files = []
    result = ""
    device = torch.device(0)
    for file in audio:
        files.append(file.name)
    pipe = pipeline(model=model)
    pipe.save_pretrained(models_dir)
    for file in files:
        transcriptions.append(pipe(file, chunk_length_s=10)['text'])
    for x in range(len(transcriptions)):
        result += (os.path.basename(files[x]) + ": " + transcriptions[x] + "\n")
    return result
