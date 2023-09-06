import os.path

import gradio as gr
from huggingface_hub import HfApi, ModelFilter
from settings import models_dir, languages


def get_models_list(language="Polish", update=True):
    models_list = []
    api = HfApi()
    try:
        models = (api.list_models(sort="downloads", direction=-1,
                                  filter=ModelFilter(
                                      task="automatic-speech-recognition",
                                      language=languages.get(language)
                                  )
                                  ))
    except ConnectionError:
        gr.Error("Couldn't connect to HuggingFace Hub. Check your internet connection.")
    for model in models:
        models_list.append(model.modelId)
    if update:
        return [gr.Dropdown.update(choices=models_list, value=models_list[0]), set_model_link(models_list[0])]
    else:
        return models_list


def set_model_link(model):
    url = "https://huggingface.co/" + model
    return gr.Button.update(link=url)
