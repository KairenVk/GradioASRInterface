import os
import gradio as gr

import mltu_function
import wav2vec2_function

model_types = ["MLTU", "wav2vec2", "whisper"]
models_dir = "Models/"
models = os.listdir(models_dir)


def transcribe_file(audio, model, model_type):
    if model_type == "MLTU":
        return mltu_function.transcribe(audio, model)
    elif model_type == "wav2vec2":
        return wav2vec2_function.transcribe(audio, model)


with gr.Blocks() as demo:
    with gr.Tab("Single File"):
        model_type = gr.Radio(label="Model type", choices=model_types)
        model = gr.Dropdown(label="Model", choices=models)
        audio = gr.Audio(label="Audio", type="filepath")
        output = gr.Textbox(label="Output Box")
        greet_btn = gr.Button("Transcribe")
        greet_btn.click(fn=transcribe_file, inputs=[audio, model, model_type], outputs=output, api_name="Transcribe")

demo.launch()
