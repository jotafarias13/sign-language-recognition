### SIGN LANGUAGE RECOGNITION ###
### preprocessing.py ###

## import modules
import numpy as np
import wandb
import matplotlib.pyplot as plt


## Downloading artifacts from wandb
print("Downloading artifacts from wandb...")
os.system("wandb login --relogin")

run = wandb.init(project="sign_language_recognition", save_code=True)

artifact_wandb_train = run.use_artifact("sign_language_recognition/raw_data.npy:latest")
artifact_wandb_labels = run.use_artifact("sign_language_recognition/raw_data_labels.npy:latest")

raw_train = artifact_wandb_train.file()
raw_train_labels = artifact_wandb_labels.file()

X = np.load(raw_train)
Y = np.load(raw_train_labels)
print("Done!")


print(f"X.shape: {X.shape}")
print(f"Y.shape: {Y.shape}")


## Visualizing 'NULL' images
Y_NULL = np.where(Y == 'NULL')[0]
print("Indices of 'NULL' images:")
print(Y_NULL)

print(f"Number of 'NULL' images: {len(Y_NULL)}")
print(f"X.shape: {X.shape}")
print(f"Y.shape: {Y.shape}")

print("Deleting 'NULL' images...")
new_X = np.delete(X, Y_NULL, axis=0)
new_Y = np.delete(Y, Y_NULL)
print(f"new_X.shape: {new_X.shape}")
print(f"new_Y.shape: {new_Y.shape}")
print("Done!")


## Finding indices of duplicate images
print("Finding indices of duplicate images...")
duplicated = 0
i_array = []
j_array = []

for i in range(len(new_X)-1):
  for j in range(i+1, len(new_X)):
    if np.array_equal(new_X[i], new_X[j]):
      duplicated += 1
      i_array.append(i)
      j_array.append(j)

print(f"duplicated: {duplicated}")
print(f"i_array: {i_array}")
print(f"j_array: {j_array}")
print(f"Number of unique values in i_array: {len(np.unique(i_array))}")
print(f"Number of unique values in j_array: {len(np.unique(j_array))}")

print("We can either delete the indices of i_array or j_array")
print(f"new_X.shape: {new_X.shape}")
print(f"new_Y.shape: {new_Y.shape}")

print("Deleting duplicate images...")
j_array = np.unique(j_array)
new_X = np.delete(new_X, j_array, axis=0)
new_Y = np.delete(new_Y, j_array)
print(f"new_X.shape: {new_X.shape}")
print(f"new_Y.shape: {new_Y.shape}")
print("Done!")


## Saving preprocessed arrays
print("Saving preprocessed arrays...")
np.save("preprocessed_data.npy", new_X)
np.save("preprocessed_data_labels.npy", new_Y)
print("Done!")
print(f"Size of preprocessed_data.npy: {os.path.getsize('preprocessed_data.npy')}")
print(f"Size of preprocessed_data_labels.npy: {os.path.getsize('preprocessed_data_labels.npy')}")


# Exporting data to wandb
artifact_train = wandb.Artifact(name="preprocessed_data.npy",
        type="preprocessed_data",
        description="Preprocessed data (npy file) from Sign Language Recognition Dataset (without labels)")
artifact_train.add_file("preprocessed_data.npy")


artifact_train_labels = wandb.Artifact(name="preprocessed_data_labels.npy",
        type="preprocessed_data",
        description= "Preprocessed data (npy file) from Sign Language Recognition Dataset (only labels)")
artifact_train_labels.add_file("preprocessed_data_labels.npy")

run.log_artifact(artifact_train)
run.log_artifact(artifact_train_labels)
run.finish()



## Deleting local files
print("Deleting local files...")
os.system("rm preprocessed_data.npy preprocessed_data_labels.npy")
print("Done!")



print("In the next step, we will deal with data checking!")
