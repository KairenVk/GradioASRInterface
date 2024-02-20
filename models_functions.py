import gradio as gr
from huggingface_hub import HfApi, ModelFilter
from modelClass import Model
from modelListClass import Model_list
from app_settings import settings_yaml


def get_models_list(language="Polish", update=True):
    models_list = Model_list([])
    api = HfApi()
    try:
        models = (api.list_models(sort="downloads", direction=-1,
                                  filter=ModelFilter(
                                      task="automatic-speech-recognition",
                                      language=settings_yaml['languages'].get(language),
                                  )
                                  ))
    except ConnectionError:
        gr.Error("Couldn't connect to HuggingFace Hub. Check your internet connection.")
    for model in models:
        if model.tags.__contains__('transformers'):
            models_list.append(Model(model.modelId, 'transformers'))
        elif model.tags.__contains__('ctranslate2'):
            models_list.append(Model(model.modelId, 'ctranslate2'))
    if update:
        refresh_models(models_list, models_list.get_model_names_list())
        return [gr.Dropdown.update(choices=hub_models_names, value=hub_models_names[0]),
                set_model_link(hub_models_names[0])]
    else:
        return models_list


def refresh_models(models: Model_list, models_names):
    global hub_models
    hub_models = models
    global hub_models_names
    hub_models_names = models_names


hub_models = get_models_list(update=False) # X -> UPDATE -> Y
hub_models_names = hub_models.get_model_names_list()


def set_model_link(model):
    url = "https://huggingface.co/" + model
    return gr.Button.update(link=url)
