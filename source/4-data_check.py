### SIGN LANGUAGE RECOGNITION ###
### data_check.py ###


## import modules
import wandb

# login to wandb
os.system("wandb login --relogin")


with open("test_data.py", "w") as fyle:
    fyle.write('''\
## import modules
import numpy as np
import wandb
import pytest

run = wandb.init(project="sign_language_recognition", job_type="data_check")

@pytest.fixture(scope="session")

def data():

    artifact_data = run.use_artifact("sign_language_recognition/preprocessed_data.npy:latest")
    artifact_labels = run.use_artifact("sign_language_recognition/preprocessed_data_labels.npy:latest")

    preprocessed_data = artifact_data.file()
    preprocessed_labels = artifact_labels.file()

    X = np.load(preprocessed_data)
    Y = np.load(preprocessed_labels)

    return [X, Y]


def test_length_X_Y(data):
    """
    Check whether X and Y have the same length
    """
    X = data[0]
    Y = data[1]
    assert len(X) == len(Y)
    
def test_RGB_channels(data):
    """
    Check whether the images are in RBG (3 channels)
    """
    X = data[0]
    assert X.shape[3] == 3

def test_RGB_range(data):
    """
    Check whether RBG is between 0 and 255
    """
    X = data[0]
    in_range = True

    for img in X:
        range_0 = np.where(img < 0)[0]
        range_255 = np.where(img > 255)[0]
        if len(range_0)!=0 or len(range_255)!=0:
            in_range = False
            break

    assert in_range == True

def test_number_images(data):
    """
    Check whether there are over 5k images
    """
    X = data[0]
    assert len(X) > 5000

def test_number_classes(data):
    """
    Check whether there are 26 classes
    """
    Y = data[1]
    unique = len(np.unique(Y))
    assert unique == 26


run.finish()
    ''')


os.system("pytest . -vv")

print("Deleting test file...")
os.system("rm test_data.py")
print("Done!")



# ## import modules
# import numpy as np
# import pytest

# run = wandb.init(project="sign_language_recognition", job_type="data_check")

# @pytest.fixture(scope="session")

# def data():

#     artifact_data = run.use_artifact("sign_language_recognition/preprocessed_data.npy")
#     artifact_labels = run.use_artifact("sign_language_recognition/preprocessed_data_labels.npy")

#     preprocessed_data = artifact_data.file()
#     preprocessed_labels = artifact_labels.file()

#     X = np.load(preprocessed_data)
#     Y = np.load(preprocessed_labels)

#     return [X, Y]


# def test_length_X_Y(data):
#     """
#     Check whether X and Y have the same length
#     """
#     X = data[0]
#     Y = data[1]
#     assert len(X) == len(Y)
    
# def test_RGB_channels(data):
#     """
#     Check whether the images are in RBG (3 channels)
#     """
#     X = data[0]
#     assert X.shape[3] == 3

# def test_RGB_range(data):
#     """
#     Check whether RBG is between 0 and 255
#     """
#     X = data[0]
#     in_range = True

#     for img in X:
#         range_0 = np.where(img < 0)[0]
#         range_255 = np.where(img > 255)[0]
#         if len(range_0)!=0 or len(range_255)!=0:
#             in_range = False
#             break

#     assert in_range == True

# def test_number_images(data):
#     """
#     Check whether there are over 5k images
#     """
#     X = data[0]
#     assert len(X) > 5000

# def test_number_classes(data):
#     """
#     Check whether there are 26 classes
#     """
#     Y = data[1]
#     unique = len(np.unique(Y))
#     assert unique == 26


# run.finish()
