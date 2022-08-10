### SIGN LANGUAGE RECOGNITION ###
### eda.py ###

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


## Visualizing data
print(f"Number of unique values of Y: {len(np.unique(Y))}")
unique_values_Y, counts_Y = np.unique(Y, return_counts=True)
print(f"Unique values: {unique_values_Y}")
print(f"Counts: {counts_Y}")
print(f"Number of 'NULL' values: {np.count_nonzero(Y == 'NULL')}")


## Visualizing 10 random images
print("Visualizing 10 random images...")
random = np.sort((np.random.rand(10)*(len(X)+1)).astype(int))

fig = plt.figure(figsize=(8,4))
fig_rows = 2
fig_columns = 5
for i in range(len(random)):
    fig.add_subplot(fig_rows, fig_columns, i+1)
    plt.imshow(X[random[i]])
    plt.axis('off')
    plt.title(Y[random[i]])
plt.show()

## Visualizing 'NULL' images
Y_NULL = np.where(Y == 'NULL')[0]
print("Indices of 'NULL' images:")
print(Y_NULL)

print("First 10 'NULL' images:")
fig = plt.figure(figsize=(8,4))
for i in range(10):
    fig.add_subplot(fig_rows, fig_columns, i+1)
    plt.imshow(X[Y_NULL[i]])
    plt.axis('off')
    plt.title(Y[Y_NULL[i]])
plt.show()

## Histogram of classes
print("Histogram of classes")
fig = plt.figure(figsize=(35,8))
plt.hist(Y, bins=len(np.unique(Y)), rwidth=0.7)
plt.show()
print("Target is balanced!")


## Checking for duplicate images
print("Checking for duplicate images...")
duplicated = 0
i_array = []
j_array = []

for i in range(len(X)-1):
  for j in range(i+1, len(X)):
    if np.array_equal(X[i], X[j]):
      duplicated += 1
      i_array.append(i)
      j_array.append(j)

print(f"duplicated: {duplicated}")
print(f"i_array: {i_array}")
print(f"j_array: {j_array}")
print(f"We can see there are {duplicated} duplicate images.")

print("Let's see 10 of these duplicate images just to make sure...")
random = np.sort((np.random.rand(10)*(len(j_array)+1)).astype(int))
fig_rows = 2
fig_columns = len(random)
fig = plt.figure(figsize=(35,8))
for i in range(len(random)):
    fig.add_subplot(fig_rows, fig_columns, i+1)
    plt.imshow(X[i_array[random[i]]])
    fig.add_subplot(fig_rows, fig_columns, i+1+len(random))
    plt.imshow(X[j_array[random[i]]])
plt.show()

print("In the next step, we will deal with 'NULL' and duplicate images!")
