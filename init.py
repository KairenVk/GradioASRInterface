import os
import sys
import torch
from app_settings import settings_yaml
from user_interface import user_interface
import models_functions

os.environ['TRANSFORMERS_CACHE'] = "Cache/"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

if not torch.cuda.is_available() and settings_yaml['use_cuda'] is True:
    print("CUDA-capable devices not found, reverting to CPU...")
    settings_yaml['use_cuda'] = False
try:
    hub_models = models_functions.get_models_list(update=False)
except ConnectionError:
    sys.exit("Couldn't connect to HuggingFace Hub. Please make sure you are connected to internet.")

user_interface.queue()
user_interface.launch()
