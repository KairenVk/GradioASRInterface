import gradio as gr
import transcription
import models_functions
import autocorrect_functions
from models_functions import hub_models_names
from app_settings import settings_yaml

autocorrect_languages = settings_yaml['autocorrect_languages'].keys()
languages = settings_yaml['languages'].keys()


def send_to_correction(text):
    return [text, gr.Tabs.update(selected=1)]


with gr.Blocks() as user_interface:
    with gr.Tabs(selected=0) as tabs:
        with gr.Tab("Transcription Correction", id=1):
            tc_language_dropdown = gr.Dropdown(label="Language",
                                               choices=autocorrect_languages, value="Polish")
            tc_text_input = gr.Textbox(label="Text",
                                       interactive=True)
            tc_transcribe_btn = gr.Button("Autocorrect Text")
            tc_output = gr.Textbox(label="Output",
                                   interactive=True)
            tc_transcribe_btn.click(fn=autocorrect_functions.autocorrect_text,
                                    inputs=[tc_text_input, tc_language_dropdown],
                                    outputs=tc_output,
                                    api_name="Autocorrect Text")
        with gr.Tab("HuggingFace Hub Models", id=0):
            with gr.Row():
                language_dropdown = gr.Dropdown(label="Language",
                                                choices=languages,
                                                value="Polish",
                                                scale=1)
                model_dropdown = gr.Dropdown(label="Model",
                                             choices=hub_models_names,
                                             value=hub_models_names[0],
                                             scale=2)
                model_info_btn = gr.Button(value="Model Info",
                                           link="https://huggingface.co/" + hub_models_names[0],
                                           scale=1)
                language_dropdown.input(fn=models_functions.get_models_list,
                                        inputs=[language_dropdown],
                                        outputs=[model_dropdown, model_info_btn])
                model_dropdown.input(fn=models_functions.set_model_link,
                                     inputs=model_dropdown,
                                     outputs=model_info_btn)
            with gr.Tab("Single File"):
                sf_audio = gr.Audio(label="Audio", type="filepath")
                sf_output = gr.Textbox(label="Output Box", lines=7, interactive=False)
                with gr.Row():
                    sf_transcribe_btn = gr.Button("Transcribe", variant='primary')
                    sf_transcribe_btn.click(fn=transcription.transcribe,
                                            inputs=[sf_audio, model_dropdown],
                                            outputs=sf_output,
                                            api_name="HFTranscribe")
                    sf_correct = gr.Button("Send to correction")
                    sf_correct.click(fn=send_to_correction, inputs=sf_output, outputs=[tc_text_input, tabs])
            with gr.Tab("Multiple Files"):
                mf_audio = gr.File(label="Audio files",
                                   file_types=['audio'],
                                   file_count='multiple')
                mf_output = gr.Textbox(label="Output Box", interactive=False)
                with gr.Row():
                    mf_transcribe_btn = gr.Button("Transcribe", variant='primary')
                    mf_transcribe_btn.click(fn=transcription.transcribe_multiple,
                                            inputs=[mf_audio, model_dropdown],
                                            outputs=mf_output,
                                            api_name="HFTranscribeMultiple")
                    mf_correct = gr.Button("Send to correction")
                    mf_correct.click(fn=send_to_correction, inputs=mf_output, outputs=[tc_text_input, tabs])
            with gr.Tab("Live Audio"):
                la_audio = gr.Audio(source="microphone",
                                    type="filepath")
                la_output = gr.Textbox(label="Output Box")
                with gr.Row():
                    la_transcribe_btn = gr.Button("Transcribe", variant='primary')
                    la_transcribe_btn.click(fn=transcription.transcribe,
                                            inputs=[la_audio, model_dropdown],
                                            outputs=la_output,
                                            api_name="HFTranscribeLive")
                    la_correct = gr.Button("Send to correction")
                    la_correct.click(fn=send_to_correction, inputs=la_output, outputs=[tc_text_input, tabs])
