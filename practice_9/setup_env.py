import subprocess
import sys

packages = ["torch", "transformers", "nltk", "datasets"]
subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

import torch
import transformers
import nltk
import datasets

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("nltk:", nltk.__version__)
print("datasets:", datasets.__version__)
