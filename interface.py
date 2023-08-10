import os
import gradio as gr

import autocorrect_functions
import settings
import hub_functions

model_types = settings.model_types
models_dir = settings.models_dir
hub_models = []
local_models = []
autocorrect_languages = []
for model in hub_functions.getModels():
    hub_models.append(model.modelId)
for key, value in settings.autocorrect_languages.items():
    autocorrect_languages.append(key)


def placeholder():
    print("test")


with gr.Blocks() as demo:
    with gr.Tab("HuggingFace Hub Models"):
        with gr.Tab("Single File"):
            model = gr.Dropdown(label="Model", choices=hub_models)
            audio = gr.Audio(label="Audio", type="filepath")
            output = gr.Textbox(label="Output Box")
            greet_btn = gr.Button("Transcribe")
            greet_btn.click(fn=hub_functions.transcribe, inputs=[audio, model], outputs=output, api_name="Transcribe")
        with gr.Tab("Multiple Files"):
            model = gr.Dropdown(label="Model", choices=hub_models)
            audio = gr.File(label="Audio files", file_types=['audio'], file_count='multiple')
            output = gr.Textbox(label="Output Box")
            greet_btn = gr.Button("Transcribe")
            greet_btn.click(fn=hub_functions.transcribe_multiple, inputs=[audio, model], outputs=output, api_name="Transcribe")
        with gr.Tab("Live Audio"):
            model = gr.Dropdown(label="Model", choices=hub_models)
            audio = gr.Audio(label="Audio", type="filepath")
            output = gr.Textbox(label="Output Box")
            greet_btn = gr.Button("Transcribe")
            greet_btn.click(fn=placeholder, inputs=[], outputs=output, api_name="Transcribe")
    with gr.Tab("Local MLTU Models"):
        with gr.Tab("Single File"):
            model = gr.Dropdown(label="Model", choices=hub_models)
            audio = gr.Audio(label="Audio", type="filepath")
            output = gr.Textbox(label="Output Box")
            greet_btn = gr.Button("Transcribe")
            greet_btn.click(fn=placeholder, inputs=[], outputs=output, api_name="Transcribe")
    with gr.Tab("Transcription Correction"):
        language = gr.Dropdown(label="Language", choices=autocorrect_languages)
        input = gr.Textbox(label="Text", interactive=True)
        greet_btn = gr.Button("Autocorrect Text")
        output = gr.Textbox(label="Output", interactive=True)
        greet_btn.click(fn=autocorrect_functions.autocorrect_text, inputs=[input, language], outputs=output, api_name="Autocorrect Text")

os.environ['TRANSFORMERS_CACHE'] = 'Models/'
demo.launch()
