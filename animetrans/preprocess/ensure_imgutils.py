"""
Dependency management module for dghs-imgutils library.

This module provides functionality to automatically detect and install the dghs-imgutils
library dependency. It ensures that the required image processing utilities are available
before the main application runs, providing a seamless user experience by handling
dependency installation automatically.

The module performs validation of specific required modules within dghs-imgutils to
ensure complete functionality is available.
"""
import subprocess
import sys
import warnings


def ensure_imgutils_dependency():
    """
    Detect and automatically install the dghs-imgutils library dependency.

    This function checks if the dghs-imgutils library is installed and accessible.
    If not found, it attempts to install the library automatically using pip.
    After installation, it validates that the required modules are properly imported.

    :raises RuntimeError: If the installation fails or the library cannot be imported after installation.

    Example::

        >>> # Ensure the dependency is available before using imgutils
        >>> ensure_imgutils_dependency()
        dghs-imgutils library is already installed and available.

        >>> # If not installed, it will automatically install
        >>> ensure_imgutils_dependency()
        dghs-imgutils library not found: No module named 'imgutils'
        Attempting to install dghs-imgutils...
        Successfully installed dghs-imgutils!
        dghs-imgutils library is now available.
    """
    try:
        exec('import imgutils')
        exec('from imgutils.preprocess import create_torchvision_transforms, parse_torchvision_transforms')
    except (ImportError, ModuleNotFoundError) as e:
        warnings.warn(f"dghs-imgutils library not found: {e}\n"
                      f"Attempting to install dghs-imgutils...")

        try:
            # Attempt to install dghs-imgutils
            subprocess.check_call([sys.executable, "-m", "pip", "install", "dghs-imgutils>=0.17.0"])
            print("Successfully installed dghs-imgutils!")

            # Re-import to verify installation
            exec('import imgutils')
            exec('from imgutils.preprocess import create_torchvision_transforms, parse_torchvision_transforms')
            print("dghs-imgutils library is now available.")
        except subprocess.CalledProcessError as install_error:
            raise RuntimeError(
                f"Failed to install dghs-imgutils: {install_error}\n"
                "Please install it manually using: pip install dghs-imgutils"
            ) from install_error
        except (ImportError, ModuleNotFoundError) as import_error:
            raise RuntimeError(
                f"dghs-imgutils was installed but cannot be imported: {import_error}\n"
                "Please check your installation or install manually: pip install dghs-imgutils"
            ) from import_error
