# Speech-to-text Transcription Interface
Gradio web interface app, designed to be used with Transformers and ctranslate2 Automatic Speech Recognition models available on HuggingFace. 
Allows to transcribe single and multiple files, as well as live audio using microphone.
Transcription outputs are saved as text files in Outputs folder.

## Setup (Windows):

For CPU users:
- Install python dependencies: `pip install -r requirements.txt`
- Download FFmpeg: https://ffmpeg.org
- Add FFmpeg directory to PATH system variable.

For Nvidia GPU users:
- Install python dependencies: `pip install -r requirements.txt`
- Install torch with GPU support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Install Nvidia CUDA Toolkit v11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Install Nvidia cuDNN library: https://developer.nvidia.com/cudnn
- Download FFmpeg: https://ffmpeg.org
- Add FFmpeg directory to PATH system variable.

## Settings

File "settings.yaml" allows user to switch between using CUDA-capable devices (such as GPU) and change directory used to store models.

## Usage

To run the app, use `start.bat` file, or run `python init.py` in Command Prompt.