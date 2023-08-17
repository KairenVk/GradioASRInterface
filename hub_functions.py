import os
import gradio as gr
from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments
import torch
from transformers import pipeline
from settings import models_dir, languages


def get_models_list(language="Polish", count=10, update=False):
    models_list = []
    api = HfApi()
    try:
        models = (api.list_models(sort="downloads", direction=-1, limit=count,
                                  filter=ModelFilter(
                                      task="automatic-speech-recognition",
                                      language=languages.get(language)
                                  )
                                  ))
    except ConnectionError:
        print("Couldn't connect to HuggingFace Hub. Check your internet connection.")
        return models_list
    for model in models:
        models_list.append(model.modelId)
    print(models_list[0])
    if update:
        return [gr.Dropdown.update(choices=models_list, value=models_list[0]), gr.Button.update(link=f"https://huggingface.co/{models_list[0]}")]
    else:
        return models_list


def set_model_link(model):
    url = "https://huggingface.co/" + model
    return gr.Button.update(link=url)


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
