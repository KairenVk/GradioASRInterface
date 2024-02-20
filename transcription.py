import os
import output_file
import gradio as gr
from faster_whisper import WhisperModel
from transformers import pipeline
from app_settings import settings_yaml


def transcribe(audio, model):
    from models_functions import hub_models
    modelInfo = hub_models.get_model_by_name(model)
    gr.Info("Preparing model...")
    if settings_yaml['use_cuda']:
        model_device = 'cuda'
    else:
        model_device = 'cpu'
    if modelInfo.type == 'ctranslate2':
        model = WhisperModel(modelInfo.name, device=model_device, download_root=settings_yaml['models_dir'],
                             compute_type="int8")
        gr.Info("Transcribing file...")
        segments, info = model.transcribe(audio)
        gr.Info("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        result = ''
        for segment in segments:
            result += segment.text
        output_file.save_output(audio, modelInfo, result)
        return result

    elif modelInfo.type == 'transformers':
        pipe = pipeline(model=modelInfo.name, device=model_device)
        pipe.save_pretrained(settings_yaml['models_dir'])
        gr.Info("Transcribing file...")
        result = pipe(audio, chunk_length_s=10)['text']
        output_file.save_output(audio, modelInfo, result)
        return result


def transcribe_multiple(audio, model):
    from models_functions import hub_models
    modelInfo = hub_models.get_model_by_name(model)
    transcriptions = []
    files = []
    result = ""
    gr.Info("Preparing model...")
    for file in audio:
        files.append(file.name)
    if modelInfo.type == 'ctranslate2':
        if settings_yaml['use_cuda']:
            model_device = 'cuda'
        else:
            model_device = 'cpu'
        model = WhisperModel(modelInfo.name, device=model_device, download_root=settings_yaml['models_dir'],
                             compute_type="int8")
        gr.Info("Transcribing file...")
        for file in files:
            file_result = ''
            segments, info = model.transcribe(file)
            for segment in segments:
                file_result += segment.text
            transcriptions.append(file_result)
        for x in range(len(transcriptions)):
            result += (os.path.basename(files[x]) + ": " + "\"" + transcriptions[x] + "\"" + "\n")
        output_file.save_multiple_output(model, result)
        return result

    elif modelInfo.type == 'transformers':
        pipe = pipeline(model=modelInfo.name)
        pipe.save_pretrained(settings_yaml['models_dir'])
        for file in files:
            transcriptions.append(pipe(file, chunk_length_s=10)['text'])
        for x in range(len(transcriptions)):
            result += (os.path.basename(files[x]) + ": " + "\"" + transcriptions[x] + "\"" + "\n")
        output_file.save_multiple_output(model, result)
        return result
