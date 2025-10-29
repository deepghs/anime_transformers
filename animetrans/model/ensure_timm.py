"""
Dependency management module for timm library.

This module provides functionality to automatically detect and install the timm
library dependency. It ensures that the required model creation utilities are available
before the main application runs, providing a seamless user experience by handling
dependency installation automatically.

The module performs validation of specific required modules within timm to
ensure complete functionality is available.
"""
import subprocess
import sys
import warnings


def ensure_timm_dependency() -> None:
    """
    Detect and automatically install the timm library dependency.

    This function checks if the timm library is installed and accessible.
    If not found, it attempts to install the library automatically using pip.
    After installation, it validates that the required modules are properly imported.

    :raises RuntimeError: If the installation fails or the library cannot be imported after installation.

    Example::

        >>> # Ensure the dependency is available before using timm
        >>> ensure_timm_dependency()
        timm library is already installed and available.

        >>> # If not installed, it will automatically install
        >>> ensure_timm_dependency()
        timm library not found: No module named 'timm'
        Attempting to install timm...
        Successfully installed timm!
        timm library is now available.
    """
    try:
        exec('import timm')
        exec('from timm import create_model as _timm_create_model')
    except (ImportError, ModuleNotFoundError) as e:
        warnings.warn(f"timm library not found: {e}\n"
                      f"Attempting to install timm...")

        try:
            # Attempt to install timm
            subprocess.check_call([sys.executable, "-m", "pip", "install", "timm>=1.0"])
            print("Successfully installed timm!")

            # Re-import to verify installation
            exec('import timm')
            exec('from timm import create_model as _timm_create_model')
            print("timm library is now available.")
        except subprocess.CalledProcessError as install_error:
            raise RuntimeError(
                f"Failed to install timm: {install_error}\n"
                "Please install it manually using: pip install timm>=1.0"
            ) from install_error
        except (ImportError, ModuleNotFoundError) as import_error:
            raise RuntimeError(
                f"timm was installed but cannot be imported: {import_error}\n"
                "Please check your installation or install manually: pip install timm>=1.0"
            ) from import_error
