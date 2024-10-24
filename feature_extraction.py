from etils import epath
from ml_collections import config_dict
import os
import tensorflow as tf
import tqdm
from chirp.inference import colab_utils
colab_utils.initialize(use_tf_gpu=True, disable_warnings=True)

from chirp import audio_utils
from chirp.inference import embed_lib
from chirp.inference import tf_examples
import librosa
import soundfile as sf
from audioread import NoBackendError
from chunk import Chunk
import aifc
import wave

# Set the root directory for Australia_ROI
root_dir = '/mnt/d/Aqoustics/BEN/Australia_ROI/'

# Define the model
model_choice = 'perch'  #@param

config = config_dict.ConfigDict()
config.embed_fn_config = config_dict.ConfigDict()
config.embed_fn_config.model_config = config_dict.ConfigDict()

# Recursive function to gather all .wav files from subdirectories with progress bar
def gather_audio_files(root_dir):
    audio_files = []
    total_files = sum([len(files) for _, _, files in os.walk(root_dir)])  # To estimate total files
    with tqdm.tqdm(total=total_files, desc="Gathering audio files") as pbar:
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.wav'):
                    full_path = os.path.join(dirpath, file)
                    audio_files.append(full_path)
                    pbar.update(1)
    print(f"Found {len(audio_files)} audio files:")
    for f in audio_files:
        print(f)
    return audio_files

config.source_file_patterns = gather_audio_files(root_dir)
config.output_dir = '/mnt/d/Aqoustics/BEN/Australia_Embeddings'

# For Perch, the directory containing the model.
perch_model_path = '/home/os/aqoustics/Aqoustics-Surfperch/kaggle'

if model_choice == 'perch':
    config.embed_fn_config.model_key = 'taxonomy_model_tf'
    config.embed_fn_config.model_config.window_size_s = 5.0
    config.embed_fn_config.model_config.hop_size_s = 5.0
    config.embed_fn_config.model_config.sample_rate = 32000
    config.embed_fn_config.model_config.model_path = perch_model_path

# Only write embeddings to reduce size.
config.embed_fn_config.write_embeddings = True
config.embed_fn_config.write_logits = False
config.embed_fn_config.write_separated_audio = False
config.embed_fn_config.write_raw_audio = False

# Number of parent directories to include in the filename.
config.embed_fn_config.file_id_depth = 1
config.tf_record_shards = 100

# Set up the embedding function, including loading models.
embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
print(f'\n\nLoading model(s) from: ', perch_model_path)
embed_fn.setup()

# Create output directory and write the configuration.
output_dir = epath.Path(config.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
embed_lib.maybe_write_config(config, output_dir)

# Create SourceInfos.
source_infos = embed_lib.create_source_infos(
    config.source_file_patterns,
    num_shards_per_file=config.get('num_shards_per_file', -1),
    shard_len_s=config.get('shard_len_s', -1))
print(f'Found {len(source_infos)} source infos.')

# Function to safely load audio files
def safe_load_audio(filepath: str, sample_rate: int):
    try:
        audio, sr = librosa.load(filepath, sr=sample_rate)
        return audio
    except (librosa.util.exceptions.ParameterError, sf.LibsndfileError, NoBackendError, EOFError, wave.Error, aifc.Error) as e:
        print(f"Skipping file {filepath}: {str(e)}")
        return None

# Set up the audio loader function
audio_loader = lambda fp, offset: safe_load_audio(fp, config.embed_fn_config.model_config.sample_rate)

# Initialize counters for successful, failed, and skipped files
succ, fail, skipped = 0, 0, 0

# Initialize audio_iterator to None
audio_iterator = None

try:
    # Use source_infos instead of new_source_infos
    audio_iterator = audio_utils.multi_load_audio_window(
        filepaths=[s.filepath for s in source_infos],
        offsets=[s.shard_num * s.shard_len_s for s in source_infos],
        audio_loader=audio_loader,
    )

    # Adding progress bar for processing source_infos
    with tqdm.tqdm(total=len(source_infos), desc="Processing embeddings") as pbar:
        with tf_examples.EmbeddingsTFRecordMultiWriter(
            output_dir=output_dir, num_files=config.get('tf_record_shards', 1)) as file_writer:
            
            for source_info, audio in zip(source_infos, audio_iterator):
                file_id = source_info.file_id(config.embed_fn_config.file_id_depth)
                offset_s = source_info.shard_num * source_info.shard_len_s
                if audio is None:
                    skipped += 1
                    pbar.update(1)
                    continue
                example = embed_fn.audio_to_example(file_id, offset_s, audio)
                if example is None:
                    fail += 1
                    pbar.update(1)
                    continue
                file_writer.write(example.SerializeToString())
                succ += 1
                pbar.update(1)
            file_writer.flush()
finally:
    # Only delete audio_iterator if it was defined
    if audio_iterator is not None:
        del (audio_iterator)

# Print summary of processing results
print(f'\n\nSuccessfully processed {succ} source_infos.')
print(f'Failed to process {fail} source_infos.')
print(f'Skipped {skipped} files due to errors.')
