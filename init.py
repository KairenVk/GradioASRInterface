import sys
import torch
import models_functions
from settings import use_cuda
from user_interface import user_interface


if not torch.cuda.is_available() and use_cuda is True:
    print("CUDA-capable devices not found, reverting to CPU...")
    use_cuda = False

try:
    hub_models = models_functions.get_models_list(update=False)
except ConnectionError:
    sys.exit("Couldn't connect to HuggingFace Hub. Please make sure you are connected to internet.")

user_interface.queue()
user_interface.launch()
