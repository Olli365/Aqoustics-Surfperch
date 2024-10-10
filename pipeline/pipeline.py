import os
import pandas as pd
import soundfile as sf
import time
from maad import sound
from maad.util import power2dB, format_features
from maad.rois import create_mask, select_rois
import numpy as np
import tensorflow as tf
import tqdm
from chirp.inference import embed_lib, tf_examples
from etils import epath
from ml_collections import config_dict  # Import added here


def process_audio(file_path, output_folder, embed_fn, tf_record_writer):
    print(f"Processing file: {file_path}")
    start_time = time.time()

    try:
        # Load the audio file
        s, fs = sound.load(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()

    try:
        # Attempt to filter the signal
        s_filt = sound.select_bandwidth(s, fs, fcut=100, forder=3, ftype='highpass')
    except ValueError as e:
        print(f"Skipping file {file_path} due to filtering error: {e}")
        return pd.DataFrame()

    # Spectrogram parameters
    db_max = 70
    Sxx, tn, fn, ext = sound.spectrogram(s_filt, fs, nperseg=1024, noverlap=512)
    Sxx_db = power2dB(Sxx, db_range=db_max) + db_max

    # Background removal and smoothing
    Sxx_db_rmbg, _, _ = sound.remove_background(Sxx_db)
    Sxx_db_smooth = sound.smooth(Sxx_db_rmbg, std=1.2)
    im_mask = create_mask(im=Sxx_db_smooth, mode_bin='relative', bin_std=2, bin_per=0.1)
    im_rois, df_rois = select_rois(im_mask, min_roi=50, max_roi=None)

    if df_rois.empty:
        print(f"No ROIs found in file: {file_path}")
        return pd.DataFrame()

    # Format ROIs
    df_rois = format_features(df_rois, tn, fn)

    # Filter ROIs for those with centroid frequency below 2000Hz
    low_freq_rois = df_rois[df_rois['max_f'] <= 2000]

    if low_freq_rois.empty:
        print(f"No low frequency ROIs found in file: {file_path}")
        return pd.DataFrame()

    # Extract start and end times of the filtered ROIs
    low_freq_timestamps = low_freq_rois[['min_t', 'max_t']]
    low_freq_timestamps.columns = ['begin', 'end']

    # Generate 5-second audio clips with detected region in the middle
    audio_clips = []
    clip_duration = 2  # seconds

    for i, (start, end) in enumerate(low_freq_timestamps.itertuples(index=False)):
        mid_point = (start + end) / 2
        clip_start = max(0, mid_point - clip_duration / 2)
        clip_end = clip_start + clip_duration

        if clip_end > len(s) / fs:
            clip_end = len(s) / fs
            clip_start = clip_end - clip_duration

        start_sample = int(clip_start * fs)
        end_sample = int(clip_end * fs)
        audio_clip = s[start_sample:end_sample]

        # Save the audio clip to a file
        clip_filename = f'clip_{os.path.basename(file_path).split(".")[0]}_{i}.wav'
        clip_path = os.path.join(output_folder, clip_filename)
        sf.write(clip_path, audio_clip, fs)
        audio_clips.append((clip_start, clip_filename))

        # Zero-pad the audio clip to make it 5 seconds long
        target_duration = 5  # target duration in seconds
        target_samples = int(target_duration * fs)
        current_samples = len(audio_clip)

        if current_samples < target_samples:
            padding_needed = target_samples - current_samples
            pad_before = padding_needed // 2
            pad_after = padding_needed - pad_before
            padded_clip = np.pad(audio_clip, (pad_before, pad_after), 'constant')
        else:
            padded_clip = audio_clip

        # Embedding creation
        if embed_fn is not None:
            embedding = embed_fn.embedding_model.embed(padded_clip)

            # Prepare TFRecord example
            file_id = f'{file_path}_{i}'
            offset_s = clip_start
            example = embed_fn.audio_to_example(file_id, offset_s, padded_clip)

            # Write the embedding to TFRecord
            if example is not None:
                tf_record_writer.write(example.SerializeToString())

    # Create DataFrame for the audio clips
    df_audio_clips = pd.DataFrame(audio_clips, columns=['start_time', 'audio_clip'])

    end_time = time.time()
    print(f"Finished processing file: {file_path}")

    return df_audio_clips


def process_folder(input_folder, output_folder, embed_fn, record_file):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_timestamps = []
    
    # Make sure the record file's parent directory exists
    record_file_dir = os.path.dirname(record_file)
    if not os.path.exists(record_file_dir):
        os.makedirs(record_file_dir)
    
    # Use the record_file as the path to the TFRecord file
    with tf_examples.EmbeddingsTFRecordMultiWriter(output_dir=record_file, num_files=1) as tf_record_writer:
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(input_folder, filename)
                timestamps = process_audio(file_path, output_folder, embed_fn, tf_record_writer)
                if not timestamps.empty:
                    timestamps['file'] = filename
                    all_timestamps.append(timestamps)

    if all_timestamps:
        # Concatenate all timestamps and save to Excel
        all_timestamps_df = pd.concat(all_timestamps, ignore_index=True)
        excel_path = os.path.join(output_folder, 'timestamps.xlsx')
        all_timestamps_df.to_excel(excel_path, index=False)
        return all_timestamps_df
    else:
        print("No audio clips generated from any files.")
        return pd.DataFrame()



# Example usage with embedding
def initialize_embedding():
    model_choice = 'perch'
    config = config_dict.ConfigDict()
    config.embed_fn_config = config_dict.ConfigDict()
    config.embed_fn_config.model_config = config_dict.ConfigDict()

    # Set model parameters for Perch
    config.embed_fn_config.model_key = 'taxonomy_model_tf'
    config.embed_fn_config.model_config.window_size_s = 5.0  # Window size in seconds
    config.embed_fn_config.model_config.hop_size_s = 2.5     # Hop size in seconds (added)
    config.embed_fn_config.model_config.sample_rate = 32000

    # Use the correct WSL path for the model
    config.embed_fn_config.model_config.model_path = '/home/os/aqoustics/Aqoustics-Surfperch/kaggle'

    config.embed_fn_config.write_embeddings = True
    config.embed_fn_config.write_logits = False
    config.embed_fn_config.write_separated_audio = False
    config.embed_fn_config.write_raw_audio = False
    config.embed_fn_config.file_id_depth = 1

    # Initialize the embedding function
    embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
    embed_fn.setup()

    return embed_fn





# Example usage
input_folder = '/mnt/d/Aqoustics/UMAP'
output_folder = "/mnt/d/Aqoustics/UMAP/test"
record_file = "/mnt/d/Aqoustics/UMAP/test/"

embed_fn = initialize_embedding()

start_time = time.time()
all_timestamps = process_folder(input_folder, output_folder, embed_fn, record_file)
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")
