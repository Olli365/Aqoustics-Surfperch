 #@title Imports. { vertical-output: true }
from etils import epath
from ml_collections import config_dict
import numpy as np
import os
import tensorflow as tf
import tqdm
from chirp.inference import colab_utils
colab_utils.initialize(use_tf_gpu=True, disable_warnings=True)

from chirp import audio_utils
from chirp.inference import embed_lib
from chirp.inference import tf_examples



#@title Configuration. { vertical-output: true }
base_dir = os.getenv('BASE_DIR')
if not base_dir:
    raise ValueError("BASE_DIR environment variable is not set.")

# Define the model
model_choice = 'perch'  #@param

config = config_dict.ConfigDict()
config.embed_fn_config = config_dict.ConfigDict()
config.embed_fn_config.model_config = config_dict.ConfigDict()

# Pick the input and output targets.
# source_file_patterns should contain a list of globs of audio files, like:
# ['/home/me/*.wav','/home/me/*.WAV', '/home/me/other/*.flac']
config.source_file_patterns = [os.path.join(base_dir,'ucl_perch/data/australia/raw_audio/*.WAV')] #@param
config.output_dir = os.path.join(base_dir,'ucl_perch/data/australia/embeddings')  #@param

# For Perch, the directory containing the model.
# Alternatively, set the perch_tfhub_model_version, and the model will load
# directly from TFHub.
# Note that only one of perch_model_path and perch_tfhub_version should be set.
perch_model_path = os.path.join(base_dir,'/marrs_acoustics/SurfPerch-model')  #@param
print(perch_model_path)