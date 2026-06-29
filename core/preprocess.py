"""
AdobeStockSVGOptimizer
core/preprocess.py

Image preprocessing module.
"""

from pathlib import Path

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Basic image preprocessing before vectorization.
    """

    def __init__(
        self,
        max_size=2048,
        remove_noise=True,
        auto_contrast=True,
        blur=True,
        blur_kernel=(3, 3),
    ):

        self.max_size = max_size
        self.remove_noise = remove_noise
        self.auto_contrast = auto_contrast
        self.blur = blur
        self.blur_kernel = blur_kernel

    def load(self, image_path):
        """
        Load image from disk.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Cannot read image : {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def resize(self, image):

        h, w = image.shape[:2]

        longest = max(h, w)

        if longest <= self.max_size:
            return image

        scale = self.max_size / longest

        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        return image

    def denoise(self, image):

        if not self.remove_noise:
            return image

        image = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            5,
            5,
            7,
            21,
        )

        return image

    def contrast(self, image):

        if not self.auto_contrast:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return image

    def smooth(self, image):

        if not self.blur:
            return image

        image = cv2.GaussianBlur(
            image,
            self.blur_kernel,
            0,
        )

        return image

    def process(self, image_path):
        """
        Full preprocessing pipeline.
        """

        image = self.load(image_path)

        image = self.resize(image)

        image = self.denoise(image)

        image = self.contrast(image)

        image = self.smooth(image)

        return image
