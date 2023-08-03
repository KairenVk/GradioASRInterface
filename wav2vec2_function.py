from huggingsound import SpeechRecognitionModel

def transcribe(audio, model):
    model = SpeechRecognitionModel("Models/wav2vec2-large-xlsr-53-polish")
    transcription = model.transcribe([audio])
    return transcription[0].get("transcription")

