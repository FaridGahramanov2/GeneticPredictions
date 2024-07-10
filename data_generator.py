from typing import Dict

import numpy as np
import pickle as pkl


def generate_dataset(data_size: int, seed: int, file_name: str, scale: float = 0.1):
    """
        This method randomly generates a dataset based on given parameters.

        :param data_size: Number of instances to be obtained
        :param seed: Random seed
        :param file_name: The file name that the data will be saved
        :param scale: Noise scale for the target value
    """
    np.random.seed(seed)

    # Randomly generate the input features
    x1 = np.random.normal(2., 1., (data_size, ))
    x2 = np.random.normal(-2., 1., (data_size, ))

    # Randomly generate the parameters
    a = np.random.normal(0., 1.)
    b = np.random.normal(0., 1.)
    c = np.random.normal(0., 1.)

    # Generate target value with a noise
    y = a * x1 + b * x2 + c + np.random.normal(0., scale, (data_size, ))

    # Save to file
    with open(file_name, "wb") as f:
        pkl.dump({
            "x1": x1,
            "x2": x2,
            "y": y
        }, f)


def load_dataset(file_name: str) -> Dict[str, np.ndarray]:
    """
        This method loads the data from the given file path. It provides the independent features and dependent target
        value as a Python dictionary.

        :param file_name: The file path where the data exists
        :return: Dataset as a Python dictionary
    """
    with open(file_name, "rb") as f:
        data = pkl.load(f)

    return data
