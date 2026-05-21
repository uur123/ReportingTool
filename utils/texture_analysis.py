import numpy as np

from skimage.feature import graycomatrix
from skimage.feature import graycoprops

from skimage.measure import shannon_entropy


def texture_features(image):

    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast')[0, 0]

    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    energy = graycoprops(glcm, 'energy')[0, 0]

    entropy = shannon_entropy(image)

    return {
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "entropy": entropy
    }
