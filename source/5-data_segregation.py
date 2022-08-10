### SIGN LANGUAGE RECOGNITION ###
### data_segregation.py ###

## import modules
import numpy as np
import wandb
import logging
from sklearn.model_selection import train_test_split


## configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s -- %(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S")

logger = logging.getLogger()

## login to wandb
os.system("wandb login --relogin")
run = wandb.init(project="sign_language_recognition", job_type="split_data")

## importing artifacts
logger.info("Downloading artifacts...")
artifact_data = run.use_artifact("sign_language_recognition/preprocessed_data.npy:latest")
artifact_labels = run.use_artifact("sign_language_recognition/preprocessed_data_labels.npy:latest")
segregated_data = artifact_data.file()
segregated_data_labels = artifact_labels.file()
logger.info("Done!")

    
## creating arrays
logger.info("Creating arrays...")
X = np.load(segregated_data)
Y = np.load(segregated_data_labels)
logger.info("Done!")


## splitting data
logger.info("Splitting data into train and test sets...")
test_size = 0.2
seed = 13
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed, shuffle=True)
logger.info("Done!")

logger.info("Checking resulting shapes...")
print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"x_test.shape: {x_test.shape}")
print(f"y_test.shape: {y_test.shape}")
logger.info("Done!")


## saving arrays locally
logger.info("Saving files locally...")
np.save("train.npy", x_train)
np.save("train_labels.npy", y_train)
np.save("test.npy", x_test)
np.save("test_labels.npy", y_test)
logger.info("Done!")

## uploading files to wandb
logger.info("Uploading files to wandb...")

artifact_train = wandb.Artifact(name="train.npy",
                                type="segregated_data",
                                description="Train data after segregation (without labels)")
artifact_train.add_file("train.npy")

artifact_train_labels = wandb.Artifact(name="train_labels.npy",
                                    type="segregated_data",
                                    description="Labels of train data after segregation")
artifact_train_labels.add_file("train_labels.npy")

artifact_test = wandb.Artifact(name="test.npy",
                                type="segregated_data",
                                description="Test data after segregation (without labels)")
artifact_test.add_file("test.npy")

artifact_test_labels = wandb.Artifact(name="test_labels.npy",
                                    type="segregated_data",
                                    description="Labels of test data after segregation")
artifact_test_labels.add_file("test_labels.npy")

run.log_artifact(artifact_train)
run.log_artifact(artifact_train_labels)
run.log_artifact(artifact_test)
run.log_artifact(artifact_test_labels)

artifact_train.wait()
artifact_train_labels.wait()
artifact_test.wait()
artifact_test_labels.wait()

logger.info("Done!")


## Finishing wandb run
logger.info("Finishing wandb run...")
run.finish()
logger.info("Done!")


## deleting local files
logger.info("Deleting local files...")
os.system("rm train.npy train_labels.npy test.npy test_labels.npy") 
logger.info("Done!")

