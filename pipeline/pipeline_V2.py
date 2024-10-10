import os
import pandas as pd
import soundfile as sf
import time
from maad import sound
from maad.util import power2dB, format_features
from maad.rois import create_mask, select_rois
import numpy as np
import tensorflow as tf
import librosa
from tqdm import tqdm
from chirp.inference import embed_lib, tf_examples
from ml_collections import config_dict  # <-- Make sure this is imported


# Function to extract 5s clips around ROIs and resample them to 32kHz
# Function to extract 5s clips around ROIs and resample them to 32kHz
def extract_5s_clips_around_rois(s, fs, rois, target_sr=32000):
    """
    Extract 5-second audio clips around ROIs, resample them to 32kHz, 
    and ensure they are exactly 5 seconds long.
    
    Parameters:
    s (numpy array): The audio signal.
    fs (int): Original sampling rate of the audio signal.
    rois (pandas DataFrame): Dataframe of ROIs with 'min_t' and 'max_t'.
    target_sr (int): Target sample rate (default is 32000).
    
    Returns:
    List of tuples: Each tuple contains (clip_start_time, resampled_audio_clip).
    """
    audio_clips = []
    clip_duration = 5  # 5-second clip duration
    num_samples_target_sr = int(clip_duration * target_sr)  # Number of samples in 5 seconds at target_sr
    
    # Resample the entire audio to the target sample rate first
    s_resampled = librosa.resample(s, orig_sr=fs, target_sr=target_sr)
    fs = target_sr  # Update fs to target_sr for further calculations

    for i, roi in rois.iterrows():
        # Get the start and end times for the ROI, and calculate the middle point
        mid_point = (roi['min_t'] + roi['max_t']) / 2
        clip_start = max(0, mid_point - clip_duration / 2)
        clip_end = clip_start + clip_duration

        # Convert times to sample indices in the resampled signal
        start_sample = int(clip_start * fs)
        end_sample = start_sample + num_samples_target_sr

        # Ensure the clip is exactly 5 seconds long by padding or trimming
        if end_sample > len(s_resampled):
            # Pad the audio if it exceeds the original signal length
            audio_clip = s_resampled[start_sample:]
            padding_needed = num_samples_target_sr - len(audio_clip)
            audio_clip = np.pad(audio_clip, (0, padding_needed), mode='constant')
        else:
            # Extract the exact 5 seconds of audio
            audio_clip = s_resampled[start_sample:end_sample]
        
        # Append the audio clip and its corresponding start time
        audio_clips.append((clip_start, audio_clip))
    
    return audio_clips


# Modified audio processing function that identifies ROIs, saves clips, and generates embeddings
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
        # Filter the signal to remove high frequencies
        s_filt = sound.select_bandwidth(s, fs, fcut=100, forder=3, ftype='highpass')
    except ValueError as e:
        print(f"Skipping file {file_path} due to filtering error: {e}")
        return pd.DataFrame()

    # Spectrogram parameters
    db_max = 70
    Sxx, tn, fn, ext = sound.spectrogram(s_filt, fs, nperseg=1024, noverlap=512)
    Sxx_db = power2dB(Sxx, db_range=db_max) + db_max

    # Background removal and ROI detection
    Sxx_db_rmbg, _, _ = sound.remove_background(Sxx_db)
    Sxx_db_smooth = sound.smooth(Sxx_db_rmbg, std=1.2)
    im_mask = create_mask(im=Sxx_db_smooth, mode_bin='relative', bin_std=2, bin_per=0.1)
    im_rois, df_rois = select_rois(im_mask, min_roi=50, max_roi=None)

    if df_rois.empty:
        print(f"No ROIs found in file: {file_path}")
        return pd.DataFrame()

    # Format ROIs and filter low-frequency ROIs
    df_rois = format_features(df_rois, tn, fn)
    low_freq_rois = df_rois[df_rois['max_f'] <= 2000]

    if low_freq_rois.empty:
        print(f"No low-frequency ROIs found in file: {file_path}")
        return pd.DataFrame()

    # Extract 5-second audio clips around the low-frequency ROIs
    audio_clips = extract_5s_clips_around_rois(s, fs, low_freq_rois)

    clip_records = []
    for i, (start_time, audio_clip) in enumerate(audio_clips):
        # Ensure no overlap or duplication
        if i > 0 and audio_clips[i-1][0] == start_time:
            print(f"Warning: Duplicate or overlapping ROI at index {i}, skipping.")
            continue
        
        # Save the audio clip as a .wav file (optional)
        clip_filename = f'clip_{os.path.basename(file_path).split(".")[0]}_{i}.wav'
        clip_path = os.path.join(output_folder, clip_filename)
        sf.write(clip_path, audio_clip, 32000)
        clip_records.append((start_time, clip_filename))

        # Embedding creation
        if embed_fn is not None:
            # Model expects batch dimension, so use np.newaxis to add it
            embedding = embed_fn.embedding_model.embed(audio_clip[np.newaxis, :])
            
            # Prepare TFRecord example (optional)
            file_id = f'{file_path}_{i}'
            offset_s = start_time
            example = embed_fn.audio_to_example(file_id, offset_s, audio_clip)

            # Write the embedding to TFRecord
            if example is not None:
                tf_record_writer.write(example.SerializeToString())

    # Create a DataFrame to store start times and filenames of the clips
    df_audio_clips = pd.DataFrame(clip_records, columns=['start_time', 'audio_clip'])

    end_time = time.time()
    print(f"Finished processing file: {file_path}")

    return df_audio_clips



# Process the entire folder of audio files
def process_folder(input_folder, output_folder, embed_fn, record_file):
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_timestamps = []

    # Ensure the directory for TFRecord file exists
    record_file_dir = os.path.dirname(record_file)
    if not os.path.exists(record_file_dir):
        os.makedirs(record_file_dir)

    # Write embeddings to TFRecord
    with tf_examples.EmbeddingsTFRecordMultiWriter(output_dir=record_file, num_files=1) as tf_record_writer:
        for filename in tqdm(os.listdir(input_folder), desc="Processing audio files"):
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


# Initialize the embedding model
# Initialize the embedding model
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

    # Set logits output to False
    config.embed_fn_config.write_embeddings = True
    config.embed_fn_config.write_logits = False  # Ensure logits are not written
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
