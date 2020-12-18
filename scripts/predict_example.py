import numpy as np
from PIL import Image

from predict import Prediction
from config import train_params, classes_mask
from utils import colorize_mask

def predict_image(model_path: str, image: np.ndarray, show: bool = False):

    pred = Prediction(model_path=model_path,
                        output_path=None,
                        patch_size=train_params['patch_size'],
                        batch_size=train_params['batch_size'],
                        offset=2*train_params["n_layers"],
                        n_classes=len(classes_mask.keys()),
                        n_filters=train_params["n_filters"],
                        n_layers=train_params["n_layers"])

    # image = np.array(Image.open(image)).astype(np.float32) / 255
    predicted = pred.__predict_image__(image)
    predicted = np.argmax(predicted, axis=2)

    if show:
        Image.fromarray(colorize_mask(np.dstack((predicted, predicted, predicted)), n_classes=pred.n_classes)).show()

    return predicted