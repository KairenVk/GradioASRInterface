import typing

import numpy as np
from mltu.configs import BaseModelConfigs
from mltu.inferenceModel import OnnxInferenceModel
from mltu.preprocessors import WavReader
from mltu.utils.text_utils import ctc_decoder


models_dir = "Models/"
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


def transcribe(audio, model):
    configs = BaseModelConfigs.load(models_dir+model+mltu_config)
    model = WavToTextModel(model_path=configs.model_path, char_list=configs.vocab, force_cpu=False)
    spectrogram = WavReader.get_spectrogram(audio, frame_length=configs.frame_length, frame_step=configs.frame_step,
                                            fft_length=configs.fft_length)
    padded_spectrogram = np.pad(spectrogram, ((configs.max_spectrogram_length - spectrogram.shape[0], 0), (0, 0)),
                                mode='constant', constant_values=0)

    text = model.predict(padded_spectrogram)
    return text
