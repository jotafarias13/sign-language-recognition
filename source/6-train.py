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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

# tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import Accuracy
from keras.callbacks import EarlyStopping

# pipeline
from pipeline_classes import FeatureSelector, NumericalTransformer

# wandbcallback
from wandb.keras import WandbCallback


## configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s -- %(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S")

logger = logging.getLogger()

## login to wandb
logger.info("Logging to wandb...")
os.system("wandb login --relogin")
run = wandb.init(project="sign_language_recognition", job_type="train")
logger.info("Done!")


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
# val_size = 0.3
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

# Transforming image arrays (pipeline processing)
x_train_copy = x_train.copy()
x_val_copy = x_val.copy()
x_train_copy = x_train_copy/255 
x_val_copy = x_val_copy/255 


# logger.info("Creating NN Model...")
# lenet5 = Sequential()
# lenet5.add(Conv2D(filters=16, kernel_size=(5,5), strides=1,  activation='relu', input_shape=x_train_copy[0].shape, padding='same'))
# lenet5.add(BatchNormalization())
# lenet5.add(MaxPooling2D(pool_size=(3,3))) 
# lenet5.add(Dropout(0.5))
# lenet5.add(Conv2D(filters=32, kernel_size=(5,5), strides=1,  activation='relu', padding='valid'))
# lenet5.add(BatchNormalization())
# lenet5.add(MaxPooling2D(pool_size=(3,3))) 
# lenet5.add(Dropout(0.5))
# lenet5.add(Flatten()) 
# lenet5.add(Dense(units=120, activation='relu'))
# lenet5.add(Dropout(0.5))
# lenet5.add(Dense(units=84, activation='relu'))
# lenet5.add(Dropout(0.25))
# lenet5.add(Dense(units=26, activation='softmax'))
# logger.info("Done!")

logger.info("Creating NN Model...")
lenet5 = Sequential()
lenet5.add(Conv2D(filters=6, kernel_size=(5,5), strides=1, kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(0.01),  activation='relu', input_shape=(80,80,3), padding='same'))
lenet5.add(BatchNormalization())
lenet5.add(MaxPooling2D(pool_size=(3,3))) 
lenet5.add(Conv2D(filters=16, kernel_size=(5,5), strides=1,  activation='relu', padding='valid'))
lenet5.add(BatchNormalization())
lenet5.add(MaxPooling2D(pool_size=(3,3))) 
lenet5.add(Dropout(0.5))
lenet5.add(Conv2D(filters=32, kernel_size=(5,5), strides=1,  activation='relu', padding='valid'))
lenet5.add(BatchNormalization())
lenet5.add(MaxPooling2D(pool_size=(3,3))) 
# lenet5.add(Dropout(0.25))
lenet5.add(Flatten()) 
lenet5.add(Dense(units=120, activation='relu'))
# lenet5.add(Dropout(0.25))
lenet5.add(Dense(units=84, activation='relu'))
lenet5.add(Dense(units=26, activation='softmax'))
logger.info("Done!")

summary = lenet5.summary()

logger.info("Compiling and Training...")
lenet5.compile(optimizer='Adam',
               loss='sparse_categorical_crossentropy', 
               metrics='accuracy')
history = lenet5.fit(x=x_train_copy,
                    y=y_train,
                    batch_size=32,
                    # batch_size=16,
                    epochs=50,
                    validation_data=(x_val_copy,y_val),
                    callbacks=[WandbCallback()])
logger.info("Done!")


logger.info("Saving model...")
lenet5.save('best_model.h5')
logger.info("Done!")



## Pipeline

## Testing functions
fs = FeatureSelector()
x_train_fs = fs.fit_transform(x_train)
nt = NumericalTransformer(normalize=True, image_height=80, image_width=80)
x_train_num = nt.fit_transform(x_train_fs)
print(f"x_train after Feature Selector:\n{x_train_fs[0]}")
print(f"x_train after Numerical Transformation:\n{x_train_num[0]}")
print(f"x_train_num.shape: {x_train_num.shape}")


## Pipeline Creation
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 80

numerical_pipeline = Pipeline(steps=[('num_selector', FeatureSelector()),
                                     ('num_transformer', NumericalTransformer(normalize=True, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH))])

full_pipeline_preprocessing = FeatureUnion(transformer_list=[('num_pipeline', numerical_pipeline)])

# Testing pipeline
new_data = full_pipeline_preprocessing.fit_transform(x_train)
print(f"Transformed x_train.shape: {new_data.shape}")
print(f"Transformed x_train: {new_data[0]}")


## Transforming train and validation sets
x_train = full_pipeline_preprocessing.fit_transform(x_train)
x_val = full_pipeline_preprocessing.transform(x_val)

print(f"New shapes:")
print(f"x_train.shape: {x_train.shape}")
print(f"x_val.shape: {x_val.shape}")

run.finish()


### Hyperparameter Tuning

def train():
    # Default values for hyper-parameters we're going to sweep over
    defaults = dict(
        epochs = 10
    )

    
    # Initialize a new wandb run
    wandb.init(project="sign_language_recognition", config=defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config


    # neural network layers    
    lenet5 = Sequential()
    lenet5.add(Conv2D(filters=6, kernel_size=(5,5), strides=1, kernel_initializer='he_uniform', kernel_regularizer=keras.regularizers.l2(0.01),  activation='relu', input_shape=(80,80,3), padding='same'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    lenet5.add(Conv2D(filters=16, kernel_size=(5,5), strides=1,  activation='relu', padding='valid'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    lenet5.add(Dropout(0.5))
    lenet5.add(Conv2D(filters=32, kernel_size=(5,5), strides=1,  activation='relu', padding='valid'))
    lenet5.add(BatchNormalization())
    lenet5.add(MaxPooling2D(pool_size=(3,3))) 
    lenet5.add(Dropout(0.25))
    lenet5.add(Flatten()) 
    lenet5.add(Dense(units=120, activation='relu'))
    lenet5.add(Dropout(0.25))
    lenet5.add(Dense(units=84, activation='relu'))
    lenet5.add(Dense(units=26, activation='softmax'))


    # testing different loss functions
    loss = 'sparse_categorical_crossentropy'

    # Instantiate an accuracy metric.
    accuracy = Accuracy()

    optimizer = Adam()

    lenet5.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) 

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5) 
    lenet5.fit(x_train, y_train, 
               batch_size=32,
               epochs=config.epochs,
               validation_data=(x_val, y_val),
               # callbacks=[es, WandbCallback()]
               callbacks=[WandbCallback()]
               )
    
# Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric et all.
sweep_config = {
    'method': 'grid', #grid, random
    'metric': {
      'name': 'val_loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'epochs': {
            'values': [10, 25, 50]
        }
    }
}


sweep_id = wandb.sweep(sweep_config, project="sign_language_recognition")

wandb.agent(sweep_id, train)



## Creating wandb run...
logger.info("Creating wandb run...")
run = wandb.init(project="sign_language_recognition", job_type="train")
logger.info("Done!")


## Downloading The Best Model
logger.info("Downloading Best Model Found with Sweeps...")
best_model = wandb.restore('model-best.h5', run_path="jotafarias/sign_language_recognition/id0ukver")
logger.info("Done!")


# Exporting artifacts to wandb
logger.info("Exporting Encoder, Pipeline and Model to wandb...")

joblib.dump(le, "target_encoder")
logger.info(f"Target Enconder: {os.path.getsize('target_encoder')} Bytes")
artifact_encoder = wandb.Artifact("target_enconder",
                                  type="inference_artifact",
                                  description="Label encoder used on target variable")
artifact_encoder.add_file("target_encoder")
run.log_artifact(artifact_encoder)

joblib.dump(full_pipeline_preprocessing, "pipeline")
logger.info(f"Pipeline: {os.path.getsize('pipeline')} Bytes")
artifact_pipeline = wandb.Artifact("pipeline",
                                   type="inference_artifact",
                                   description="A full pipeline with proprocessing of images")
artifact_pipeline.add_file("pipeline")
run.log_artifact(artifact_pipeline)

logger.info(f"Model: {os.path.getsize(best_model.name)} Bytes")
artifact_model = wandb.Artifact("model.h5",
                                type="inference_artifact",
                                description="Model used to fit the data")
artifact_model.add_file(best_model.name)
run.log_artifact(artifact_model)

logger.info("Done!")

run.finish()


logger.info("Deleting local files...")
os.system("rm target_encoder pipeline")
logger.info("Done!")
