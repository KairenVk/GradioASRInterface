# Speech-to-text Transcription Interface
Gradio web interface app, designed to be used with Transformers and ctranslate2 Automatic Speech Recognition models available on HuggingFace. 
Allows to transcribe single and multiple files, as well as live audio using microphone.
Transcription outputs are saved as text files in Outputs folder.

## Setup (Linux):

### Tested using Python 3.11.7.

For CPU users:
- Run `./setup.sh`,
- Install FFmpeg for your distro if necessary, `sudo pacman -S ffmpeg`.

For GPU users:
- Run `./setup.sh`,
- Install CUDNN package for your distro, eg. `sudo pacman -S cudnn`,
- Install FFmpeg for your distro if necessary, `sudo pacman -S ffmpeg`.

## Setup (Windows):

### Tested using Python 3.10.

For CPU users:
- Install python dependencies: `pip install -r requirements.txt`,
- Download FFmpeg: https://ffmpeg.org,
- Add FFmpeg directory to PATH system variable.

For Nvidia GPU users:
- Install python dependencies: `pip install -r requirements.txt`,
- Install torch with GPU support: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`,
- Install Nvidia CUDA Toolkit v12.1: https://developer.nvidia.com/cuda-11-8-0-download-archive,
- Install Nvidia cuDNN library: https://developer.nvidia.com/cudnn,
- Download FFmpeg: https://ffmpeg.org,
- Add FFmpeg directory to PATH system variable.

## Settings

File "settings.yaml" allows user to switch between using CUDA-capable devices (such as GPU) and change directory used to store models.

## Usage

To run the app, use `start.bat` file (Windows),
or `source start.sh` (Linux).