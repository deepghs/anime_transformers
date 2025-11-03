"""
This module provides utilities for manipulating metadata in SafeTensors files.

SafeTensors is a format for storing tensors safely and efficiently. This module
specifically focuses on adding or updating metadata information in existing
SafeTensors files while preserving the original tensor data.
"""

from typing import Dict

from safetensors import safe_open
from safetensors.torch import save_file


def add_metadata_to_safetensors(input_path: str, output_path: str, new_metadata: Dict[str, str]) -> None:
    """
    Read a SafeTensors file, add new metadata information, and save to a new file.

    This function loads an existing SafeTensors file, preserves all tensor data,
    merges the existing metadata with new metadata (where new metadata takes
    precedence for duplicate keys), and saves the result to a new file.

    :param input_path: Path to the input SafeTensors file to read from.
    :type input_path: str
    :param output_path: Path where the output SafeTensors file will be saved.
    :type output_path: str
    :param new_metadata: Dictionary containing new metadata key-value pairs to add.
                        If keys already exist in the original metadata, they will be overwritten.
    :type new_metadata: Dict[str, str]
    :return: None
    :rtype: None
    :raises FileNotFoundError: If the input file does not exist.
    :raises PermissionError: If there are insufficient permissions to read/write files.
    :raises ValueError: If the input file is not a valid SafeTensors file.

    Example:
        >>> metadata = {
        ...     "model_name": "my_model",
        ...     "version": "1.0.0",
        ...     "author": "John Doe"
        ... }
        >>> add_metadata_to_safetensors("input.safetensors", "output.safetensors", metadata)
    """
    # Read tensor data from the original file
    tensors = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        # Get original metadata
        original_metadata = f.metadata() or {}

        # Read all tensors
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Merge metadata (new metadata will overwrite existing metadata with same keys)
    merged_metadata = {**original_metadata, **new_metadata}

    # Save to new file
    save_file(tensors, output_path, metadata=merged_metadata)


# Usage example
if __name__ == "__main__":
    # New metadata to add
    new_metadata = {
        "model_name": "my_custom_model",
        "version": "1.0.0",
        "author": "Your Name",
        "description": "Modified model with additional metadata",
        "creation_date": "2024-01-01"
    }

    # Call the function
    add_metadata_to_safetensors(
        input_path="input_model.safetensors",
        output_path="output_model.safetensors",
        new_metadata=new_metadata
    )
