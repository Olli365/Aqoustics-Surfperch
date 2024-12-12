import umap
import os # for handling files and directories
import librosa # for audio processing
import tensorflow as tf # for machine learning
import tensorflow_hub as hub # for machine learning
import numpy as np # for numerical processing
import pandas as pd # for handling dataframes
from tqdm import tqdm # for progress bar
import matplotlib.pyplot as plt # for potting
from sklearn.cluster import KMeans # for clustering
from chirp.inference import tf_examples


# To get reproducabel results, we set the seed
random_seed = 42
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Path to the directory containing the TFRecord files
tfrecord_dir  = '/mnt/d/Aqoustics/BEN/Kenya/Kenya_Embeddings/'




def list_files_in_folder(folder_path):
    """
    Returns a list of all files in the given folder path.

    Parameters:
    folder_path (str): The path to the folder.

    Returns:
    List[str]: A list of file names in the folder.
    """
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    # List all files in the folder (excluding directories)
    files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    return files




def process_in_batches(tfrecord_files, batch_size=5):
    """
    Process TFRecord files in batches and return a single DataFrame with all the embeddings.
    
    Parameters:
    tfrecord_files (list): List of paths to the TFRecord files.
    batch_size (int): Number of files to process in each batch.
    
    Returns:
    pd.DataFrame: DataFrame containing the embeddings and filenames from all batches.
    """
    # Initialize an empty list to store the dataframes from each batch
    df_list = []
    
    # Loop over the files in batches
    for i in range(0, len(tfrecord_files), batch_size):
        # Select the current batch of files
        batch_files = tfrecord_files[i:i + batch_size]
        
        print(f"Processing batch {i//batch_size + 1}")
        
        # Process the current batch and get a DataFrame
        batch_df = read_embeddings_to_dataframe(batch_files)
        
        # Append the batch dataframe to the list
        df_list.append(batch_df)
    
    # Concatenate all batch dataframes into one
    final_df = pd.concat(df_list, ignore_index=True)
    
    return final_df


def read_embeddings_to_dataframe(tfrecord_files):
    """
    Read the embeddings from the TFRecord files and return them as a DataFrame.
    
    Parameters:
    tfrecord_files (list): List of paths to the TFRecord files.
    
    Returns:
    pd.DataFrame: DataFrame containing the embeddings and filenames.
    """
    # Initialize empty lists to store filenames and embeddings
    filenames = []
    embeddings = []

    # Loop through the list of TFRecord files
    for tfrecord_file in tfrecord_files:
        # Create a TFRecordDataset from the current TFRecord file
        ds = tf.data.TFRecordDataset(tfrecord_file)

        # Use the example parser from tf_examples to parse the embeddings
        parser = tf_examples.get_example_parser()
        ds = ds.map(parser)

        # Iterate through the dataset and extract filenames and embeddings
        for ex in ds.as_numpy_iterator():
            filename = ex['filename'].decode("utf-8")  # Decode the byte string
            embedding = ex['embedding'].flatten()  # Flatten the embedding for easier handling
            filenames.append(filename)
            embeddings.append(embedding)

    # Convert the embeddings list to a DataFrame
    df = pd.DataFrame(embeddings)
    
    # Add the filenames as a separate column
    df['filename'] = filenames

    # Reorder columns to have 'filename' first
    df = df[['filename'] + [col for col in df.columns if col != 'filename']]

    return df


# Path to the directory containing the TFRecord files
tfrecord_dir = '/mnt/d/Aqoustics/BEN/Kenya/Kenya_Embeddings/'

# List the TFRecord files in the directory
file_list = list_files_in_folder(tfrecord_dir)
if 'reduced_feature_embeddings.csv' in file_list:
    file_list.remove('reduced_feature_embeddings.csv')
if 'config.json' in file_list:
    file_list.remove('config.json')

# Construct the full paths to the TFRecord files
tfrecord_files = [os.path.join(tfrecord_dir, f) for f in file_list]

# Process the TFRecord files in batches of 5
embeddings_df = process_in_batches(tfrecord_files, batch_size=5)

print(f"Embeddings DataFrame created with shape: {embeddings_df.shape}")


# Example: Saving the features metadata after processing in batches
features_df = embeddings_df
"""
def extract_metadata_from_filename(file):
    # Split the filename using 'clip_ind_' as the delimiter
    parts = file.split('_')
    
    # Extract the first letter after 'clip_ind_' to determine the class_type
    class_type = parts[1][0] if len(parts) > 1 else None
    
    return class_type
"""
def extract_metadata_from_filename(file):
    # Split the filename using '_' as the delimiter
    parts = file.split('_')
    
    # Extract the first part (before the first '_') and get its first letter(s)
    first_part = parts[0] if len(parts) > 0 else None
    
    return first_part


# Applying the function to each filename in the DataFrame
features_df['class_type'] = features_df['filename'].apply(extract_metadata_from_filename)

# Arrange columns in the desired order
# Here, embedding columns are likely represented by integers rather than starting with 'feature_'
column_order = ['filename', 'class_type'] + \
               [col for col in features_df.columns if isinstance(col, int)]  # Adjust this to match your actual column names

# Reorder the DataFrame
features_metadata_df = features_df[column_order]



import umap
n_neighbors = 13
min_dist = 0.1

#sampled_data = features_metadata_df.sample(frac=0.05, random_state=random_seed)
sampled_data = features_metadata_df.sample(frac=0.5, random_state=random_seed)

umap_reducer = umap.UMAP(n_components=128, random_state=random_seed, n_neighbors=n_neighbors, min_dist=min_dist)
umap_reducer.fit(sampled_data.iloc[:, 2:])
reduced_features_128 = umap_reducer.transform(features_metadata_df.iloc[:, 2:])
print("reduced to 128")
umap_reducer_2d = umap.UMAP(n_components=2, random_state=random_seed, n_neighbors=n_neighbors, min_dist=min_dist)
reduced_features_2d = umap_reducer_2d.fit_transform(reduced_features_128)
print("reduced to 2")
# Mapping from single letters to words for descriptive labels
#class_mapping = {'H': 'Healthy', 'D': 'Degraded', 'R': 'Restored', 'N': 'Newly-Restored'}]class_mapping = {'H': 'Healthy', 'D': 'Degraded', 'R': 'Restored', 'N': 'Newly-Restored'}
class_mapping = {'Healthy': 'Healthy', 'Degraded': 'Degraded', 'Restored': 'Restored'}
color_mapping = {'Healthy': 'green', 'Degraded': 'red', 'Restored': 'blue', 'Newly-Restored': 'yellow'}

# Set up the plot
plt.figure(figsize=(10, 10))

# Plot each class with its own color and label using the mapping
for class_type, label in class_mapping.items():
    # Select only data rows with the current class_type, mapping them to descriptive labels on-the-fly
    indices = features_metadata_df['class_type'] == class_type
    plt.scatter(reduced_features_2d[indices, 0], reduced_features_2d[indices, 1], label=label,
                color=color_mapping[label], alpha=0.5)  # Assigning specific colors

plt.title('UMAP Projection of Audio Features')
plt.legend(title='Class Label')  # Adds a legend with a title

# Save the plot as an image file in the specified directory
plt.savefig('/mnt/d/Aqoustics/BEN/Kenya/Kenya_UMAP/umap1.png', dpi=300)  # Adjust dpi for quality if needed
# Optionally, if you want to close the plot to free up memory:
plt.close()
print("Saved umap1.png")



from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# Step 1: Fit GMM on the 2D UMAP embeddings
n_components = 100  # Set the number of clusters (you can adjust this based on your data)
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_seed)
gmm.fit(reduced_features_2d)

# Step 2: Predict the cluster labels for each point
gmm_labels = gmm.predict(reduced_features_2d)

# Step 3: Function to draw ellipses representing GMM components
def draw_ellipse(position, covariance, ax=None, cluster_num=None, **kwargs):
    """Draw an ellipse with a given position and covariance and label it with the cluster number."""
    ax = ax or plt.gca()

    # Compute the eigenvalues and eigenvectors to find the axes of the ellipse
    if covariance.shape == (2, 2):
        eigvals, eigvecs = np.linalg.eigh(covariance)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * np.sqrt(eigvals)
    else:
        raise ValueError("Covariance matrix must be 2x2 for 2D data.")

    # Draw the Ellipse with angle as a keyword argument
    for nsig in range(1, 4):  # Draw ellipses at 1, 2, and 3 standard deviations
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle=angle, **kwargs))

    # Add the cluster number at the center of the ellipse
    if cluster_num is not None:
        ax.text(position[0], position[1], str(cluster_num), color='black', fontsize=12, ha='center', va='center')


# Step 4: Plot UMAP embeddings with clusters from GMM
plt.figure(figsize=(10, 10))

# Plot UMAP embeddings and color them by the GMM cluster labels
plt.scatter(reduced_features_2d[:, 0], reduced_features_2d[:, 1], c=gmm_labels, cmap='viridis', s=50, alpha=0.7)

# Step 5: Plot GMM ellipses with cluster numbers
w_factor = 0.2 / gmm.weights_.max()  # Scale the ellipses by their weights
for i, (pos, covar, w) in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
    # Add this print statement before calling draw_ellipse
    draw_ellipse(pos, covar, alpha=w * w_factor, color='black', cluster_num=i)  # Add cluster number

plt.title('UMAP Projection of Audio Features with GMM Clustering and Cluster Labels')
plt.colorbar(label='GMM Cluster')
plt.savefig('/mnt/d/Aqoustics/BEN/Kenya/Kenya_UMAP/umap2.png', dpi=300)  # Adjust dpi for quality if needed
plt.close()
print("Saved umap2.png")

features_metadata_df['cluster'] = gmm_labels



import os
import shutil
import pandas as pd
import random

def organize_clips_by_cluster(df, source_base_folder, destination_base_folder, n_samples):
    
    # Create the destination base folder if it doesn't exist
    if not os.path.exists(destination_base_folder):
        os.makedirs(destination_base_folder)
    
    # Iterate through each cluster in the DataFrame
    for cluster in df['cluster'].unique():
        # Filter the DataFrame for the current cluster
        cluster_df = df[df['cluster'] == cluster]
        
        # Randomly select `n_samples` files from the current cluster
        selected_files = cluster_df.sample(min(n_samples, len(cluster_df)), random_state=42)
        
        # Create a folder for the current cluster
        cluster_folder = os.path.join(destination_base_folder, f"cluster_{cluster}")
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        
        # Iterate through the selected files and copy them
        for _, row in selected_files.iterrows():
            source_file = os.path.join(source_base_folder, row['filename'])
            
            # Get the folder name the file is in
            folder_name = os.path.basename(os.path.dirname(source_file))
            
            # Create a new file name with the folder name as a prefix
            new_file_name = f"{folder_name}_{os.path.basename(row['filename'])}"
            destination_file = os.path.join(cluster_folder, new_file_name)
            
            # Copy the file to the appropriate cluster folder
            shutil.copy2(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")



source_base_folder = "/mnt/d/Aqoustics/BEN/Kenya/Kenya_ROI/"
destination_base_folder = "/mnt/d/Aqoustics/BEN/Kenya/Kenya_Clusters/"
n_samples = 100
organize_clips_by_cluster(features_metadata_df, source_base_folder, destination_base_folder,n_samples)