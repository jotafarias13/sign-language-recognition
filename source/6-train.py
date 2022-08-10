### SIGN LANGUAGE RECOGNITION ###
### train.py ###

## import modules
import numpy as np
import wandb
import logging


## login to wandb
os.system("wandb login --relogin")

run = wandb.init(project="sign_language_recognition", job_type="train")

## configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S")

logger = logging.getLogger()

logger.info("Downloading and reading artifacts from wandb...")
artifact_data = run.use_artifact("sign_language_recognition/train.npy:latest")
artifact_labels = run.use_artifact("sign_language_recognition/train_labels.npy:latest")























run.finish()
