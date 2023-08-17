import os
import gradio as gr

import autocorrect_functions
import settings
import hub_functions

model_types = settings.model_types
models_dir = settings.models_dir
hub_models = hub_functions.get_models_list()
local_models = []
autocorrect_languages = settings.autocorrect_languages.values()
languages = settings.languages.keys()


def placeholder():
    print("test")


with gr.Blocks() as demo:
    with gr.Tab("HuggingFace Hub Models"):
        with gr.Tab("Single File"):
            language = gr.Dropdown(label="Language", choices=languages, value='Polish')
            count = gr.Slider(5, 20, 10, step=1, label="Models count")
            update = gr.Checkbox(value=True, visible=False)
            with gr.Row():
                model = gr.Dropdown(label="Model", choices=hub_models, value=hub_models[0])
                model_info_btn = gr.Button(value="Model Info", link="https://huggingface.co/", size="sm")
                language.input(fn=hub_functions.get_models_list,
                               inputs=[language, count, update],
                               outputs=[model, model_info_btn])
                model.input(fn=hub_functions.set_model_link, inputs=model, outputs=model_info_btn)
            audio = gr.Audio(label="Audio", type="filepath")
            output = gr.Textbox(label="Output Box")
            transcribe_btn = gr.Button("Transcribe")
            transcribe_btn.click(fn=hub_functions.transcribe, inputs=[audio, model], outputs=output, api_name="HFTranscribe")
        with gr.Tab("Multiple Files"):
            model = gr.Dropdown(label="Model", choices=hub_models, value=hub_models[0])
            audio = gr.File(label="Audio files", file_types=['audio'], file_count='multiple')
            output = gr.Textbox(label="Output Box")
            transcribe_btn = gr.Button("Transcribe")
            transcribe_btn.click(fn=hub_functions.transcribe_multiple, inputs=[audio, model], outputs=output,
                                 api_name="HFTranscribeMultiple")
        with gr.Tab("Live Audio"):
            model = gr.Dropdown(label="Model", choices=hub_models)
            audio = gr.Audio(label="Audio", type="filepath")
            output = gr.Textbox(label="Output Box")
            transcribe_btn = gr.Button("Transcribe")
            transcribe_btn.click(fn=placeholder, inputs=[], outputs=output, api_name="Transcribe")
    with gr.Tab("Local MLTU Models"):
        with gr.Tab("Single File"):
            model = gr.Dropdown(label="Model", choices=hub_models)
            audio = gr.Audio(label="Audio", type="filepath")
            output = gr.Textbox(label="Output Box")
            transcribe_btn = gr.Button("Transcribe")
            transcribe_btn.click(fn=placeholder, inputs=[], outputs=output, api_name="Transcribe")
    with gr.Tab("Transcription Correction"):
        language = gr.Dropdown(label="Language", choices=autocorrect_languages)
        input = gr.Textbox(label="Text", interactive=True)
        transcribe_btn = gr.Button("Autocorrect Text")
        output = gr.Textbox(label="Output", interactive=True)
        transcribe_btn.click(fn=autocorrect_functions.autocorrect_text, inputs=[input, language], outputs=output,
                             api_name="Autocorrect Text")

os.environ['TRANSFORMERS_CACHE'] = 'Models/'
demo.launch()
