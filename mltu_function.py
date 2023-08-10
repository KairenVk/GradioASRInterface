import os
import typing

import gradio
import numpy as np
from mltu.configs import BaseModelConfigs
from mltu.inferenceModel import OnnxInferenceModel
from mltu.preprocessors import WavReader
from mltu.utils.text_utils import ctc_decoder
import settings

models_dir = settings.models_dir
mltu_config = "/configs.yaml"


class WavToTextModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list


    def predict(self, data: np.ndarray):
        data_pred = np.expand_dims(data, axis=0)

        preds = self.model.run(None, {self.input_name: data_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text


def transcribe_single(audio, model):
    try:
        configs = BaseModelConfigs.load(models_dir + model + mltu_config)
    except FileNotFoundError:
        raise gradio.Error("Wrong model type selected!")

    model = WavToTextModel(model_path=configs.model_path, char_list=configs.vocab, force_cpu=False)
    spectrogram = WavReader.get_spectrogram(audio, frame_length=configs.frame_length, frame_step=configs.frame_step,
                                            fft_length=configs.fft_length)
    padded_spectrogram = np.pad(spectrogram, ((configs.max_spectrogram_length - spectrogram.shape[0], 0), (0, 0)),
                                mode='constant', constant_values=0)

    text = model.predict(padded_spectrogram)
    return text


def transcribe_multiple(audio,model):
    try:
        configs = BaseModelConfigs.load(models_dir + model + mltu_config)
    except FileNotFoundError:
        raise gradio.Error("Wrong model type selected!")
    model = WavToTextModel(model_path=configs.model_path, char_list=configs.vocab, force_cpu=False)
    transcription = ""
    for file in audio:
        spectrogram = WavReader.get_spectrogram(file.name, frame_length=configs.frame_length, frame_step=configs.frame_step,
                                                fft_length=configs.fft_length)
        padded_spectrogram = np.pad(spectrogram, ((configs.max_spectrogram_length - spectrogram.shape[0], 0), (0, 0)),
                                    mode='constant', constant_values=0)

        text = model.predict(padded_spectrogram)
        transcription += ("'"+os.path.basename(file.name)+"': '"+text+"'\n")
    return transcription
