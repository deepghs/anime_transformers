"""
This module provides functionality for creating enhanced confusion matrix visualizations.

The module contains utilities for plotting confusion matrices with various customization options,
including normalization, color mapping, sampling for large datasets, and detailed metric displays.
It's designed to provide comprehensive visual analysis of classification model performance.
"""

from typing import Literal, Optional, List, Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix


def plt_confusion_matrix(ax: Axes, y_true: Union[List, np.ndarray], y_pred: Union[List, np.ndarray],
                         labels: List[str], title: str = 'Confusion Matrix',
                         normalize: Literal['true', 'pred', None] = None,
                         cmap=None, max_samples: Optional[int] = None,
                         show_values: bool = True, show_percentages: bool = True) -> Axes:
    """
    Plot an enhanced confusion matrix with comprehensive visualization features.

    This function creates a detailed confusion matrix visualization with customizable
    normalization, color mapping, and metric displays. It includes features like
    automatic sampling for large datasets, accuracy calculation, and visual enhancements
    such as highlighted diagonal elements and grid lines.

    :param ax: The matplotlib axes object to plot on.
    :type ax: matplotlib.axes.Axes
    :param y_true: True labels for the classification task.
    :type y_true: Union[List, np.ndarray]
    :param y_pred: Predicted labels from the classification model.
    :type y_pred: Union[List, np.ndarray]
    :param labels: List of class label names for display on axes.
    :type labels: List[str]
    :param title: Title for the confusion matrix plot.
    :type title: str
    :param normalize: Normalization method for the confusion matrix values.
                     'true' normalizes over true labels, 'pred' over predicted labels,
                     None shows raw counts.
    :type normalize: Literal['true', 'pred', None]
    :param cmap: Custom colormap for the heatmap. If None, uses a default grayscale colormap.
    :type cmap: matplotlib colormap, optional
    :param max_samples: Maximum number of samples to use for computation efficiency.
                       If dataset is larger, random sampling is applied.
    :type max_samples: Optional[int]
    :param show_values: Whether to display numerical values in each cell of the matrix.
    :type show_values: bool
    :param show_percentages: Whether to display percentage values alongside the main values.
    :type show_percentages: bool

    :return: The modified axes object with the plotted F1 curves.
    :rtype: matplotlib.axes.Axes

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> y_true = [0, 1, 2, 0, 1, 2]
        >>> y_pred = [0, 1, 1, 0, 2, 2]
        >>> labels = ['Class A', 'Class B', 'Class C']
        >>> plt_confusion_matrix(ax, y_true, y_pred, labels)
        >>> plt.show()
    """

    # Convert to numpy arrays for improved efficiency
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # If sample size is too large, perform random sampling
    sampled_count = None
    if max_samples is not None and len(y_true) > max_samples:
        print(f"Large dataset detected ({len(y_true)} samples). Sampling {max_samples} for efficiency.")
        indices = np.random.choice(len(y_true), max_samples, replace=False)
        sampled_count = max_samples
        y_true = y_true[indices]
        y_pred = y_pred[indices]

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    cm_counts = confusion_matrix(y_true, y_pred, normalize=None)  # Raw counts

    # Set default colormap
    if cmap is None:
        # Create more aesthetically pleasing gradient colors
        colors = ['#f7f7f7', '#cccccc', '#969696', '#636363', '#252525']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom_blues', colors, N=n_bins)

    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')

    # Set ticks and labels
    n_classes = len(labels)
    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0, va='center')

    # Set axis labels
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')

    # Add numerical annotations
    if show_values:
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                # Choose text color
                text_color = "white" if cm[i, j] > thresh else "black"

                # Prepare display text
                if normalize is None:
                    # Display raw counts
                    text = f'{int(cm[i, j])}'
                    if show_percentages and cm_counts.sum() > 0:
                        pct = cm_counts[i, j] / cm_counts.sum() * 100
                        text += f'\n({pct:.1f}%)'
                else:
                    # Display normalized values
                    text = f'{cm_counts[i, j]}'
                    if show_percentages:
                        pct = cm[i, j] * 100
                        text += f'\n({pct:.1f}%)'

                ax.text(j, i, text, ha='center', va='center',
                        color=text_color, fontsize=10, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if normalize == 'true':
        cbar.set_label('True Positive Rate', rotation=270, labelpad=20, fontsize=11)
    elif normalize == 'pred':
        cbar.set_label('Positive Predictive Value', rotation=270, labelpad=20, fontsize=11)
    else:
        cbar.set_label('Number of Samples', rotation=270, labelpad=20, fontsize=11)

    # Calculate accuracy and other metrics
    accuracy = np.trace(cm_counts) / np.sum(cm_counts)

    # Calculate precision, recall, F1 score for each class
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for i in range(n_classes):
        tp = cm_counts[i, i]
        fp = np.sum(cm_counts[:, i]) - tp
        fn = np.sum(cm_counts[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    # Set title (including key metrics)
    normalize_str = f" ({normalize} normalized)" if normalize else ""
    full_title = f"{title}{normalize_str}\n"
    if sampled_count is not None:
        full_title += f"Accuracy: {accuracy:.3f} | Samples: {sampled_count:,}"
    else:
        full_title += f"Accuracy: {accuracy:.3f}"

    ax.set_title(full_title, fontsize=14, fontweight='bold', pad=20)

    # Beautify grid
    ax.set_xlim(-0.5, n_classes - 0.5)
    ax.set_ylim(n_classes - 0.5, -0.5)

    # Add grid lines
    for i in range(n_classes + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
        ax.axvline(i - 0.5, color='white', linewidth=2)

    # Highlight diagonal (correct predictions)
    for i in range(n_classes):
        rect = plt.Rectangle((i - 0.4, i - 0.4), 0.8, 0.8,
                             fill=False, edgecolor='gold', linewidth=3, alpha=0.8)
        ax.add_patch(rect)

    return ax
