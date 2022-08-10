### SIGN LANGUAGE RECOGNITION ###
### fetch_data.py ###

## import modules
import numpy as np
import wandb
from PIL import Image

# modules necessary to download zip file
# and unzip it
import io
import zipfile

# Using kaggle API to downlaod dataset
# You need to download kaggle with "pip install kaggle"
# And authenticate the token according to Kaggle API guides
# https://www.kaggle.com/docs/api
print("Downloading dataset using Kaggle API...")
os.system("kaggle datasets download -d ardamavi/27-class-sign-language-dataset")
print("Download finished!")

# Extracting zip file
print("Extracting zip file...")
zfile = zipfile.ZipFile("27-class-sign-language-dataset.zip")
zfile.extractall("../data")
print("Extraction finished!")

# Deleting downloaded zip file
print("Deleting zip file downloaded from Kaggle...")
os.system("rm 27-class-sign-language-dataset.zip")
print("Finished!")



## Using numpy to open dataset
print("Creating numpy arrays...")
X = np.load("../data/X.npy")
Y = np.load("../data/Y.npy")
print(f"X.shape: {X.shape}")
print(f"Y.shape: {Y.shape}")
print("Finished!")

## Visualizing data
print(f"Number of unique values of Y: {len(np.unique(Y))}")
unique_values_Y, counts_Y = np.unique(Y, return_counts=True)
print(f"Unique values: {unique_values_Y}")
print(f"Counts: {counts_Y}")
print(f"Number of 'NULL' values: {np.count_nonzero(Y == 'NULL')}")




## Cutting our dataset
cut = 0.3

index = []
for i in range(len(X)):
    if np.random.rand() >= cut:
        index.append(i)

index = np.array(index)

new_X = np.delete(X, index, axis=0)
new_Y = np.delete(Y, index)

print(f"X.shape: {X.shape}")
print(f"Y.shape: {Y.shape}")
print(f"new_X.shape: {new_X.shape}")
print(f"new_Y.shape: {new_Y.shape}")

# Reshape images for size purposes
print("Reshaping images to 80x80...")
reshaped_X = []
for img in new_X:
    img = (img*255).astype(np.uint8)
    image = Image.fromarray(img)
    image = image.resize((80,80))
    img = np.array(image)
    reshaped_X.append(img)
reshaped_X = np.array(reshaped_X)
print("Done!")

print(f"reshaped_X.shape: {reshaped_X.shape}")

np.save("raw_data.npy", reshaped_X)
np.save("raw_data_labels.npy", new_Y)
print(f"Size of raw_data.npy: {os.path.getsize('raw_data.npy')}")
print(f"Size of raw_data_labels.npy: {os.path.getsize('raw_data_labels.npy')}")




# Exporting data to wandb
os.system("wandb login --relogin")
# initiate a run, syncing all steps taken on the notebook with wandb
run = wandb.init(project="sign_language_recognition", save_code=True)

artifact_train = wandb.Artifact(name="raw_data.npy",
                        type="raw_data",
                        description="Raw data (npy file) from Sign Language Recognition Dataset (without labels)")
artifact_train.add_file("raw_data.npy")


artifact_train_labels = wandb.Artifact(name="raw_data_labels.npy",
                        type="raw_data",
                        description= "Raw data (npy file) from Sign Language Recognition Dataset (only labels)")
artifact_train_labels.add_file("raw_data_labels.npy")

run.log_artifact(artifact_train)
run.log_artifact(artifact_train_labels)
run.finish()


## Deleting local files
print("Deleting local files...")
os.system("rm -r ../data")
os.system("rm raw_data.npy raw_data_labels.npy")
print("Done!")














