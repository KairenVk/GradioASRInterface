# Speech-to-text Transcription Interface
Gradio web interface app, designed to be used with Transformers and ctranslate2 Automatic Speech Recognition models available on HuggingFace. 
Allows to transcribe single and multiple files, as well as live audio using microphone.
Transcription outputs are saved as text files in Outputs folder.
---
## Setup (Linux):

### Tested using Python 3.11.7.

For CPU users:
- Run `./setup.sh`,
- Install FFmpeg for your distro if necessary, `sudo pacman -S ffmpeg`.

For GPU users:
- Run `./setup.sh`,
- Install CUDNN package for your distro, eg. `sudo pacman -S cudnn`,
- Install FFmpeg for your distro if necessary, `sudo pacman -S ffmpeg`.
---
## Setup (Windows):

### Tested using Python 3.10.

For CPU users:
- Run `setup_windows.bat`
- Download FFmpeg: https://ffmpeg.org,
- Add FFmpeg directory to PATH system variable.

For Nvidia GPU users:
- Run `setup_windows.bat'
- Install Microsoft Visual C++ Build Tools: `https://visualstudio.microsoft.com/visual-cpp-build-tools/`
    - Download and run installer,
    - Select the “C++ build tools” option and proceed with the installation.
- Install Nvidia CUDA Toolkit v12.1: https://developer.nvidia.com/cuda-11-8-0-download-archive,
- Install Nvidia cuDNN library: https://developer.nvidia.com/cudnn,
- Download FFmpeg: https://ffmpeg.org,
- Add FFmpeg directory to PATH system variable.
---
## Settings

File "settings.yaml" allows user to switch between using CUDA-capable devices (such as GPU) and change directory used to store models.

---
## Usage

To run the app, use `start.bat` file on Windows,
or `source start.sh` in terminal on Linux.