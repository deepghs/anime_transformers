"""
This module provides an image processor implementation based on imgutils library.

The module defines ImgutilsBasedImageProcessor, which extends transformers' BaseImageProcessor
to leverage imgutils preprocessing capabilities for image transformation pipelines. It supports
torchvision-style transforms and integrates seamlessly with the transformers library ecosystem.

The processor automatically handles image batching, applies configurable transformation stages,
and returns properly formatted BatchFeature objects compatible with transformers models.
"""

from typing import List, Optional, Union

from .ensure_imgutils import ensure_imgutils_dependency

# Ensure imgutils dependency is available before importing other dependencies
ensure_imgutils_dependency()

import torch

exec('from imgutils.preprocess import create_torchvision_transforms, parse_torchvision_transforms')
from transformers import BaseImageProcessor, TensorType
from transformers.image_processing_base import BatchFeature
from transformers.image_utils import make_flat_list_of_images, ImageInput


class ImgutilsBasedImageProcessor(BaseImageProcessor):
    """
    Image processor implementation based on imgutils library.

    This processor extends transformers' BaseImageProcessor to provide image preprocessing
    capabilities using imgutils. It supports configurable transformation stages defined
    as torchvision-style transforms and handles batch processing of images.

    The processor is designed to integrate seamlessly with the transformers AutoImageProcessor
    system and can be used with various vision models.

    :param stages: List of transformation stage configurations
    :type stages: List[dict]
    :param kwargs: Additional keyword arguments passed to parent class

    Example::

        >>> stages = [
        ...     {"type": "Resize", "size": [224, 224]},
        ...     {"type": "ToTensor"},
        ...     {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        ... ]
        >>> processor = ImgutilsBasedImageProcessor(stages=stages)
        >>> # Process single image or batch of images
        >>> result = processor.preprocess(images)
    """

    _auto_class = "AutoImageProcessor"

    def __init__(self, stages: List[dict], **kwargs):
        """
        Initialize the ImgutilsBasedImageProcessor.

        :param stages: List of transformation stage configurations in dictionary format
        :type stages: List[dict]
        :param kwargs: Additional keyword arguments passed to the parent BaseImageProcessor
        """
        super().__init__(**kwargs)
        # noinspection PyUnresolvedReferences
        self.stages = create_torchvision_transforms(stages)

    def preprocess(self, images: ImageInput, return_tensors: Optional[Union[str, TensorType]] = None,
                   **kwargs) -> BatchFeature:
        """
        Preprocess input images using the configured transformation stages.

        This method handles both single images and batches of images, applying the
        configured transformation pipeline to each image and returning a BatchFeature
        object containing the processed pixel values.

        :param images: Input images to preprocess. Can be a single image or batch of images
        :type images: ImageInput
        :param return_tensors: Format of the returned tensors (e.g., 'pt' for PyTorch)
        :type return_tensors: Optional[Union[str, TensorType]]
        :param kwargs: Additional keyword arguments (currently unused)

        :return: BatchFeature containing processed pixel values
        :rtype: BatchFeature

        Example::

            >>> from PIL import Image
            >>> image = Image.open("example.jpg")
            >>> result = processor.preprocess(image, return_tensors="pt")
            >>> pixel_values = result.pixel_values  # Shape: [1, C, H, W]
        """
        images = self.fetch_images(images)
        images = make_flat_list_of_images(images)
        values = []
        for image in images:
            values.append(self.stages(image))

        images = torch.stack(values)
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def to_dict(self) -> dict:
        """
        Convert the processor configuration to a dictionary representation.

        This method serializes the processor's configuration, including the transformation
        stages, into a dictionary format that can be saved and later used to reconstruct
        the processor.

        :return: Dictionary representation of the processor configuration
        :rtype: dict

        Example::

            >>> config_dict = processor.to_dict()
            >>> # Save configuration for later use
            >>> import json
            >>> with open("processor_config.json", "w") as f:
            ...     json.dump(config_dict, f)
        """
        # noinspection PyUnresolvedReferences
        return {
            **super().to_dict(),
            'stages': parse_torchvision_transforms(self.stages),
        }


# Register the processor for automatic loading
ImgutilsBasedImageProcessor.register_for_auto_class()
