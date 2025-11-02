"""
This module provides custom image augmentation transforms for training deep learning models.

It includes probabilistic grayscale conversion, image compression simulation, and a customizable
RandAugment implementation with fine-grained control over augmentation parameters. These transforms
are designed to improve model robustness by simulating real-world image variations and degradations.
"""

import io
import random
from typing import Tuple, Dict, Optional, List

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from imgutils.data import load_image


class ProbabilisticGrayscale:
    """
    Probabilistically convert images to grayscale while preserving the original color mode.

    This transform randomly converts color images to grayscale with a specified probability,
    maintaining the original image mode (RGB/RGBA) by duplicating the grayscale channel.
    This is useful for training models to be robust to color variations.
    """

    def __init__(self, p: float = 0.1):
        """
        Initialize the ProbabilisticGrayscale transform.

        :param p: Probability of applying grayscale conversion.
        :type p: float
        """
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply probabilistic grayscale conversion to the input image.

        :param img: Input PIL image to transform.
        :type img: Image.Image

        :return: Transformed image (grayscale or original).
        :rtype: Image.Image
        :raises ValueError: If the image mode is not RGB or RGBA.
        """
        if img.mode not in {'RGB', 'RGBA'}:
            raise ValueError(f'Unsupported mode, only RGB and RGBA are supported - {img!r}.')

        if random.random() < self.p:
            # Convert to grayscale but maintain original mode
            grayscale = img.convert('L')
            if img.mode == 'RGB':
                return Image.merge('RGB', (grayscale, grayscale, grayscale))
            elif img.mode == 'RGBA':
                alpha = img.split()[-1]
                return Image.merge('RGBA', (grayscale, grayscale, grayscale, alpha))

        return img

    def __repr__(self) -> str:
        """
        Return string representation of the transform.

        :return: String representation.
        :rtype: str
        """
        return f"{self.__class__.__name__}(p={self.p})"


_DEFAULT_COMP_FORMATS = ['JPEG', 'WebP']


class ImageCompression:
    """
    Probabilistically apply image compression using JPEG or WebP formats.

    This transform simulates real-world image compression artifacts by randomly applying
    lossy compression with varying quality levels. This helps models become more robust
    to compression artifacts commonly found in web images and mobile photography.
    """

    def __init__(self, p: float = 0.3, quality_range: Tuple[int, int] = (70, 95),
                 formats: Optional[List[str]] = None):
        """
        Initialize the ImageCompression transform.

        :param p: Probability of applying compression.
        :type p: float
        :param quality_range: Range of compression quality (min, max).
        :type quality_range: Tuple[int, int]
        :param formats: List of compression formats to use. Defaults to ['JPEG', 'WebP'].
        :type formats: Optional[List[str]]
        """
        self.p = p
        self.quality_range = quality_range
        self.formats = formats or _DEFAULT_COMP_FORMATS

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply probabilistic image compression to the input image.

        :param img: Input PIL image to compress.
        :type img: Image.Image

        :return: Compressed image or original image.
        :rtype: Image.Image
        """
        if random.random() < self.p:
            # Randomly select compression format and quality
            format_type = random.choice(self.formats)
            quality = random.randint(*self.quality_range)

            # Use memory buffer for compression to avoid disk I/O
            buffer = io.BytesIO()

            # Handle transparency channels
            if img.mode in ('RGBA', 'LA') and format_type == 'JPEG':
                # JPEG doesn't support transparency, convert to RGB
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background

            img.save(buffer, format=format_type, quality=quality, optimize=True)
            buffer.seek(0)
            return Image.open(buffer).copy()
        return img

    def __repr__(self) -> str:
        """
        Return string representation of the transform.

        :return: String representation.
        :rtype: str
        """
        format_str = f"formats={self.formats}" if self.formats != _DEFAULT_COMP_FORMATS else f"formats={_DEFAULT_COMP_FORMATS}"
        return (f"{self.__class__.__name__}(p={self.p}, "
                f"quality_range={self.quality_range}, {format_str})")


class WeakRandAugment2(T.RandAugment):
    """
    A customizable RandAugment implementation with fine-grained control over augmentation parameters.

    This class extends torchvision's RandAugment to provide more control over individual
    augmentation ranges and the ability to include/exclude specific augmentations.
    It's designed for scenarios where you need weaker or more controlled augmentations
    compared to the standard RandAugment implementation.
    """

    def __init__(
            self,
            num_ops: int = 2,
            magnitude: int = 9,
            num_magnitude_bins: int = 31,
            interpolation: T.InterpolationMode = T.InterpolationMode.NEAREST,
            fill: Optional[list] = None,
            # Custom upper limit ranges for each augmentation
            shear_range: float = 0.3,
            translate_range: float = 0.2,
            rotate_range: float = 30.0,
            brightness_range: float = 0.9,
            color_range: float = 0.9,
            contrast_range: float = 0.9,
            sharpness_range: float = 0.9,
            posterize_bits: int = 4,  # Posterize bit range
            # Control included/excluded augmentations
            include: Optional[List[str]] = None,
            exclude: Optional[List[str]] = None,
    ):
        """
        Initialize the WeakRandAugment2 transform.

        :param num_ops: Number of augmentation operations to apply.
        :type num_ops: int
        :param magnitude: Magnitude of augmentations (0-30).
        :type magnitude: int
        :param num_magnitude_bins: Number of magnitude bins for discretization.
        :type num_magnitude_bins: int
        :param interpolation: Interpolation mode for geometric transforms.
        :type interpolation: T.InterpolationMode
        :param fill: Fill color for areas outside the image after geometric transforms.
        :type fill: Optional[list]
        :param shear_range: Maximum shear angle range.
        :type shear_range: float
        :param translate_range: Maximum translation range as fraction of image size.
        :type translate_range: float
        :param rotate_range: Maximum rotation angle in degrees.
        :type rotate_range: float
        :param brightness_range: Maximum brightness adjustment range.
        :type brightness_range: float
        :param color_range: Maximum color saturation adjustment range.
        :type color_range: float
        :param contrast_range: Maximum contrast adjustment range.
        :type contrast_range: float
        :param sharpness_range: Maximum sharpness adjustment range.
        :type sharpness_range: float
        :param posterize_bits: Number of bits for posterize operation.
        :type posterize_bits: int
        :param include: List of augmentation names to include. If None, all are included.
        :type include: Optional[List[str]]
        :param exclude: List of augmentation names to exclude. Defaults to ['Solarize'].
        :type exclude: Optional[List[str]]
        """
        # Store custom range parameters
        self.shear_range = shear_range
        self.translate_range = translate_range
        self.rotate_range = rotate_range
        self.brightness_range = brightness_range
        self.color_range = color_range
        self.contrast_range = contrast_range
        self.sharpness_range = sharpness_range
        self.posterize_bits = posterize_bits

        # Handle include/exclude logic
        self.include = include
        self.exclude = exclude

        if self.exclude is None:
            self.exclude = ['Solarize']

        # Call parent class initialization
        super().__init__(
            num_ops=num_ops,
            magnitude=magnitude,
            num_magnitude_bins=num_magnitude_bins,
            interpolation=interpolation,
            fill=fill
        )

    def _get_full_augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[
        str, Tuple[torch.Tensor, bool]]:
        """
        Get the complete augmentation space with custom parameter ranges.

        :param num_bins: Number of magnitude bins.
        :type num_bins: int
        :param image_size: Size of the input image (height, width).
        :type image_size: Tuple[int, int]

        :return: Dictionary mapping augmentation names to (magnitudes, signed) tuples.
        :rtype: Dict[str, Tuple[torch.Tensor, bool]]
        """
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, self.shear_range, num_bins), True),
            "ShearY": (torch.linspace(0.0, self.shear_range, num_bins), True),
            "TranslateX": (torch.linspace(0.0, self.translate_range * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, self.translate_range * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, self.rotate_range, num_bins), True),
            "Brightness": (torch.linspace(0.0, self.brightness_range, num_bins), True),
            "Color": (torch.linspace(0.0, self.color_range, num_bins), True),
            "Contrast": (torch.linspace(0.0, self.contrast_range, num_bins), True),
            "Sharpness": (torch.linspace(0.0, self.sharpness_range, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / self.posterize_bits)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        """
        Get the filtered augmentation space based on include/exclude parameters.

        :param num_bins: Number of magnitude bins.
        :type num_bins: int
        :param image_size: Size of the input image (height, width).
        :type image_size: Tuple[int, int]

        :return: Filtered dictionary of augmentations.
        :rtype: Dict[str, Tuple[torch.Tensor, bool]]
        """
        # Get complete augmentation space
        full_space = self._get_full_augmentation_space(num_bins, image_size)

        # Filter based on include/exclude logic
        if self.include is not None:
            # If include is specified, filter to only include those items
            filtered_space = {k: v for k, v in full_space.items() if k in self.include}
        else:
            # If no include specified, use all augmentations
            filtered_space = full_space.copy()

        # If exclude is also specified, remove excluded items
        if self.exclude is not None:
            filtered_space = {k: v for k, v in filtered_space.items() if k not in self.exclude}

        return filtered_space

    def __repr__(self) -> str:
        """
        Return string representation of the transform.

        :return: String representation with all non-default parameters.
        :rtype: str
        """
        # Build parameter string list
        params = []

        # Basic parameters
        params.append(f"num_ops={self.num_ops}")
        params.append(f"magnitude={self.magnitude}")

        # Parameters shown only when non-default
        if self.num_magnitude_bins != 31:
            params.append(f"num_magnitude_bins={self.num_magnitude_bins}")
        if self.interpolation != T.InterpolationMode.NEAREST:
            params.append(f"interpolation={self.interpolation}")
        if self.fill is not None:
            params.append(f"fill={self.fill}")

        # Custom range parameters (shown only when non-default)
        if self.shear_range != 0.3:
            params.append(f"shear_range={self.shear_range}")
        if self.translate_range != 0.2:
            params.append(f"translate_range={self.translate_range}")
        if self.rotate_range != 30.0:
            params.append(f"rotate_range={self.rotate_range}")
        if self.brightness_range != 0.9:
            params.append(f"brightness_range={self.brightness_range}")
        if self.color_range != 0.9:
            params.append(f"color_range={self.color_range}")
        if self.contrast_range != 0.9:
            params.append(f"contrast_range={self.contrast_range}")
        if self.sharpness_range != 0.9:
            params.append(f"sharpness_range={self.sharpness_range}")
        if self.posterize_bits != 4:
            params.append(f"posterize_bits={self.posterize_bits}")

        # include/exclude parameters
        if self.include is not None:
            params.append(f"include={self.include}")
        if self.exclude != ['Solarize']:  # Default value is ['Solarize']
            params.append(f"exclude={self.exclude}")

        # Join parameter list into string
        params_str = ", ".join(params)

        return f"{self.__class__.__name__}({params_str})"


def create_augmentation(
        prob_grayscale: float = 0.0,
        prob_compress: float = 0.2,
        compress_quality_range: Tuple[int, int] = (50, 95),
        compress_formats: Optional[List[str]] = None,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: T.InterpolationMode = T.InterpolationMode.NEAREST,
        fill: Optional[list] = None,
        # Custom upper limit ranges for each augmentation
        shear_range: float = 0.3,
        translate_range: float = 0.2,
        rotate_range: float = 30.0,
        brightness_range: float = 0.9,
        color_range: float = 0.9,
        contrast_range: float = 0.9,
        sharpness_range: float = 0.9,
        posterize_bits: int = 4,  # Posterize bit range
        # Control included/excluded augmentations
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
) -> T.Compose:
    """
    Create a comprehensive image augmentation pipeline combining multiple transforms.

    This function creates a composition of transforms including probabilistic grayscale
    conversion, image compression simulation, and customizable RandAugment. It provides
    a convenient way to set up a complete augmentation pipeline for training robust
    computer vision models.

    :param prob_grayscale: Probability of applying grayscale conversion.
    :type prob_grayscale: float
    :param prob_compress: Probability of applying image compression.
    :type prob_compress: float
    :param compress_quality_range: Range of compression quality (min, max).
    :type compress_quality_range: Tuple[int, int]
    :param compress_formats: List of compression formats to use.
    :type compress_formats: Optional[List[str]]
    :param num_ops: Number of RandAugment operations to apply.
    :type num_ops: int
    :param magnitude: Magnitude of RandAugment operations (0-30).
    :type magnitude: int
    :param num_magnitude_bins: Number of magnitude bins for discretization.
    :type num_magnitude_bins: int
    :param interpolation: Interpolation mode for geometric transforms.
    :type interpolation: T.InterpolationMode
    :param fill: Fill color for areas outside the image after geometric transforms.
    :type fill: Optional[list]
    :param shear_range: Maximum shear angle range for RandAugment.
    :type shear_range: float
    :param translate_range: Maximum translation range as fraction of image size.
    :type translate_range: float
    :param rotate_range: Maximum rotation angle in degrees.
    :type rotate_range: float
    :param brightness_range: Maximum brightness adjustment range.
    :type brightness_range: float
    :param color_range: Maximum color saturation adjustment range.
    :type color_range: float
    :param contrast_range: Maximum contrast adjustment range.
    :type contrast_range: float
    :param sharpness_range: Maximum sharpness adjustment range.
    :type sharpness_range: float
    :param posterize_bits: Number of bits for posterize operation.
    :type posterize_bits: int
    :param include: List of augmentation names to include in RandAugment.
    :type include: Optional[List[str]]
    :param exclude: List of augmentation names to exclude from RandAugment.
    :type exclude: Optional[List[str]]

    :return: Composed transform pipeline.
    :rtype: T.Compose
    """
    trans = []
    if prob_grayscale > 0:
        trans.append(ProbabilisticGrayscale(p=prob_grayscale))
    if prob_compress > 0:
        trans.append(ImageCompression(
            p=prob_compress,
            quality_range=compress_quality_range,
            formats=compress_formats,
        ))
    trans.append(WeakRandAugment2(
        num_ops=num_ops,
        magnitude=magnitude,
        num_magnitude_bins=num_magnitude_bins,
        interpolation=interpolation,
        fill=fill,
        shear_range=shear_range,
        translate_range=translate_range,
        rotate_range=rotate_range,
        brightness_range=brightness_range,
        color_range=color_range,
        contrast_range=contrast_range,
        sharpness_range=sharpness_range,
        posterize_bits=posterize_bits,
        include=include,
        exclude=exclude,
    ))
    return T.Compose(trans)


if __name__ == '__main__':
    # aug = WeakRandAugment2(include=["ShearX", "ShearY"])
    aug = create_augmentation(
        prob_grayscale=0.5,
        prob_compress=0.5,
        include=['ShearX', 'ShearY'],
    )
    print(aug)
    image = load_image('test_image.jpg', mode='RGB', force_background='white')

    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
    for i in range(3):
        for j in range(3):
            ax[i, j].imshow(aug(image))

    plt.show()
