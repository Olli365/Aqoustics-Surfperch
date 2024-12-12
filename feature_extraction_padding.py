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
import numpy as np
import matplotlib.pyplot as plt

# Set the root directory for audio files
root_dir = '/mnt/d/Aqoustics/BEN/Maldives/Maldives_ROI/'

# Define the model
model_choice = 'perch'

config = config_dict.ConfigDict()
config.embed_fn_config = config_dict.ConfigDict()
config.embed_fn_config.model_config = config_dict.ConfigDict()

# Recursive function to gather all .wav files from subdirectories with progress bar
def gather_audio_files(root_dir):
    audio_files = []
    total_files = sum([len(files) for _, _, files in os.walk(root_dir)])  # Estimate total files
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
config.output_dir = '/mnt/d/Aqoustics/BEN/Maldives/Maldives_Embeddings'

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

# Function to safely load audio files and keep only the first 2 seconds and last 3 seconds of a 5-second clip
def safe_load_audio(filepath: str, sample_rate: int, clip_duration: int = 5, pad_duration_start: int = 1, pad_duration_end: int = 2):
    try:
        # Load the audio with the specified duration
        audio, sr = librosa.load(filepath, sr=sample_rate, duration=clip_duration)
        
        # Ensure audio is exactly clip_duration long (truncate or zero-pad if needed)
        audio_duration_samples = clip_duration * sample_rate
        if len(audio) < audio_duration_samples:
            # Zero-pad to make the audio exactly 5 seconds
            audio = np.pad(audio, (0, audio_duration_samples - len(audio)), mode='constant')
        else:
            # Truncate to exactly 5 seconds
            audio = audio[:audio_duration_samples]
        
        # Extract the middle section (2nd and 3rd seconds)
        middle_section = audio[sample_rate:3 * sample_rate]
        
        # Create the final padded audio with zeros in the 1st second and last 2 seconds
        padded_audio = np.concatenate((np.zeros(pad_duration_start * sample_rate), middle_section, np.zeros(pad_duration_end * sample_rate)))
        
        return padded_audio
    except (librosa.util.exceptions.ParameterError, sf.LibsndfileError, NoBackendError, EOFError, wave.Error, aifc.Error) as e:
        print(f"Skipping file {filepath}: {str(e)}")
        return None


# Test the padding on a single audio file before the main loop
sample_audio_file = config.source_file_patterns[0]  # Take the first audio file from the gathered list
sample_audio = safe_load_audio(sample_audio_file, config.embed_fn_config.model_config.sample_rate)

# Plot the original and padded audio waveforms for the first audio file
if sample_audio is not None:
    time_axis = np.linspace(0, len(sample_audio) / config.embed_fn_config.model_config.sample_rate, num=len(sample_audio))
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, sample_audio)
    plt.title("Padded Audio Waveform (Last 3 Seconds Zero-Padded, First 2 Seconds Retained)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.show()

# Set up the audio loader function for the main loop
audio_loader = lambda fp, offset: safe_load_audio(fp, config.embed_fn_config.model_config.sample_rate)

# Initialize counters for successful, failed, and skipped files
succ, fail, skipped = 0, 0, 0

# Initialize audio_iterator to None
audio_iterator = None

try:
    # Use source_infos for audio iterator
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