### SIGN LANGUAGE RECOGNITION ###
### test.py ###

## import modules
import numpy as np
import wandb
import logging
import joblib
import matplotlib.pyplot as plt

# scikit-learn
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

# imbalanced-learn
from imblearn.metrics import geometric_mean_score

# tensorflow
from tensorflow import keras

# pipeline
from pipeline_classes import FeatureSelector, NumericalTransformer


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
artifact_test = run.use_artifact("sign_language_recognition/test.npy:latest")
artifact_test_labels = run.use_artifact("sign_language_recognition/test_labels.npy:latest")
test = artifact_test.file()
test_labels = artifact_test_labels.file()
logger.info("Done!")


## Creating arrays
logger.info("Creating array...")
x_test = np.load(test)
y_test = np.load(test_labels)
logger.info("Done!")

logger.info("Checking shapes...")
print(f"x_test.shape: {x_test.shape}")
print(f"y_test.shape: {y_test.shape}")
logger.info("Done!")


## Downloading and encoding target variable
logger.info("Downloading target encoder...")
encoder = run.use_artifact("sign_language_recognition/target_enconder:latest")
encoder = encoder.file()
le = joblib.load(encoder) 
logger.info("Done!")


logger.info("Encoding target variable...")

y_test = le.transform(y_test)

start = 0
num = len(np.unique(y_test))
stop = num - 1
classes = (np.linspace(start=start, stop=stop, num=num)).astype(int)
string_classes = str(classes)
classes_names = le.inverse_transform(classes)

print("Classes:")
print(f"{string_classes}")
print(f"{classes_names}")

logger.info("Done!")


## Downloading model
logger.info("Downloading model...")
best_model = run.use_artifact("sign_language_recognition/model.h5:latest")
best_model = best_model.file()
model = keras.models.load_model(best_model)
logger.info("Done!")


## Downloading pipeline
logger.info("Downloading pipeline...")
pipeline = run.use_artifact("sign_language_recognition/pipeline:latest")
pipeline = pipeline.file()
pipeline = joblib.load(pipeline)
logger.info("Done!")


## Passing images through pipeline
logger.info("Passing data through pipeline...")
x_test = pipeline.transform(x_test)
logger.info("Done!")

print(x_test.shape)
print(x_test[0])


## Infering
logger.info("Infering...")
predict = model.predict(x_test)
# Reshaping prediction
predict_new = []
for pred in predict:
  predict_new.append(pred.argmax())
predict = np.array(predict_new)
logger.info("Done!")

logger.info("Test Evaluation Metrics:")
acc = accuracy_score(y_test, predict)
fbeta = fbeta_score(y_test, predict, beta=1, average='weighted', zero_division=1)
precision = precision_score(y_test, predict, average='weighted', zero_division=1)
recall = recall_score(y_test, predict, average='weighted', zero_division=1)
g_mean = geometric_mean_score(y_test, predict)

logger.info(f"Test Accuracy: {acc}")
logger.info(f"Test Precision: {precision}")
logger.info(f"Test Recall: {recall}")
logger.info(f"Test F1: {fbeta}")
logger.info(f"Test G_Mean: {g_mean}")

run.summary["Accuracy"] = acc
run.summary["Precision"] = precision
run.summary["Recall"] = recall
run.summary["F1"] = fbeta
run.summary["G_Mean"] = g_mean


# Classification report
logger.info("Classification Report:")
print(classification_report(y_test, predict))


# Confusion Matrix
logger.info("Confusion Matrix:")
fig_confusion_matrix, ax = plt.subplots(figsize=(12,8))
ConfusionMatrixDisplay(confusion_matrix(predict, y_test, labels=classes),
                       display_labels=classes_names).plot(values_format=".0f",ax=ax)
ax.set_xlabel("True Label")
ax.set_ylabel("Predicted Label")
plt.show()


run.finish()
