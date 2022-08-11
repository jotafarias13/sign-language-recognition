### SIGN LANGUAGE RECOGNITION ###
### train.py ###

## import modules
import numpy as np
import wandb
import logging
import joblib

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import Accuracy

# wandbcallback
from wandb.keras import WandbCallback


## configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s -- %(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S")

logger = logging.getLogger()

## login to wandb
os.system("wandb login --relogin")
run = wandb.init(project="sign_language_recognition", job_type="train")


## Downloading artifacts
logger.info("Downloading and reading artifacts from wandb...")
artifact_train = run.use_artifact("sign_language_recognition/train.npy:latest")
artifact_train_labels = run.use_artifact("sign_language_recognition/train_labels.npy:latest")
train = artifact_train.file()
train_labels = artifact_train_labels.file()
logger.info("Done!")


## Creating arrays
logger.info("Creating array...")
x_train = np.load(train)
y_train = np.load(train_labels)
logger.info("Done!")


## Splitting data
logger.info("Splitting data into train and validation sets")
val_size = 0.2
seed = 13
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                    test_size=val_size,
                                                    random_state=seed,
                                                    shuffle=True)
logger.info("Done!")

logger.info("Checking resulting shapes...")
print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"x_val.shape: {x_val.shape}")
print(f"y_val.shape: {y_val.shape}")
logger.info("Done!")


## Encoding target variable
logger.info("Encoding target variable...")

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)

start = 0
num = len(np.unique(y_train))
stop = num - 1
classes = (np.linspace(start=start, stop=stop, num=num)).astype(int)
string_classes = str(classes)
classes_names = le.inverse_transform(classes)

print("Classes:")
print(f"{string_classes}")
print(f"{classes_names}")

logger.info("Done!")


## Base model training
logger.info("Base Model Training")

x_train_copy = x_train.copy()
x_val_copy = x_val.copy()

x_train_copy = x_train_copy/255 
x_val_copy = x_val_copy/255 


logger.info("Creating NN Model...")
lenet5 = Sequential()
lenet5.add(Conv2D(6, (5,5), strides=1,  activation='tanh', input_shape=(80,80,3), padding='same')) 
lenet5.add(AveragePooling2D()) 
lenet5.add(Conv2D(16, (5,5), strides=1, activation='tanh', padding='valid')) 
lenet5.add(AveragePooling2D()) 
lenet5.add(Flatten()) 
lenet5.add(Dense(120, activation='tanh'))
lenet5.add(Dropout(0.5))
lenet5.add(Dense(84, activation='tanh'))
lenet5.add(Dropout(0.5))
lenet5.add(Dense(43, activation='softmax'))
logger.info("Done!")


logger.info("Compiling and Training...")
lenet5.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy', 
               metrics='accuracy')
history = lenet5.fit(x=x_train_copy,
                    y=y_train,
                    batch_size=32,
                    epochs=10,
                    validation_data=(x_val_copy,y_val),
                    callbacks=[WandbCallback()])
logger.info("Done!")


logger.info("Saving model...")
lenet5.save('best_model.h5')
logger.info("Done!")


# Exporting artifacts to wandb
logger.info("Exporting Encoder and Model to Wandb...")

joblib.dump(le, "target_encoder")
artifact_encoder = wandb.Artifact("target_enconder",
                                  type="inference_artifact",
                                  description="Label encoder used on target variable")
artifact_encoder.add_file("target_encoder")
run.log_artifact(artifact_encoder)

artifact_model = wandb.Artifact("best_model",
                                type="inference_artifact",
                                description="Model used to fit the data")
artifact_model.add_file("best_model.h5")
run.log_artifact(artifact_model)

logger.info("Done!")

run.finish()



logger.info("Deleting local files...")
os.system("rm best_model.h5 target_encoder")
logger.info("Done!")
