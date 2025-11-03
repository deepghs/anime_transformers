"""
This module provides functions for plotting multi-class classification metrics including 
Precision-Recall curves and F1 score curves. It is designed to visualize the performance 
of multi-class classification models with efficient sampling for large datasets.
"""

from typing import List

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


def plt_multiclass_metrics(ax: matplotlib.axes.Axes, y_true: np.ndarray, y_scores: np.ndarray,
                           labels: List[str], title: str = 'Multi-class P/R/F1 Curves',
                           max_samples: int = 50000, n_thresholds: int = 100) -> matplotlib.axes.Axes:
    """
    Plot multi-class Precision-Recall curves with micro and macro averages.

    This function creates comprehensive Precision-Recall curves for multi-class classification
    tasks, including individual class curves, micro-average, macro-average, and baseline.
    For large datasets, it performs intelligent sampling to maintain computational efficiency
    while preserving the overall distribution characteristics.

    :param ax: The matplotlib axes object for plotting.
    :type ax: matplotlib.axes.Axes
    :param y_true: True labels as class indices.
    :type y_true: np.ndarray
    :param y_scores: Prediction probability scores for each class.
    :type y_scores: np.ndarray
    :param labels: List of class label names for legend display.
    :type labels: List[str]
    :param title: Title for the plot.
    :type title: str
    :param max_samples: Maximum number of samples to use for efficiency control.
    :type max_samples: int
    :param n_thresholds: Number of thresholds for curve smoothness control.
    :type n_thresholds: int
    :return: The modified axes object with the plotted curves.
    :rtype: matplotlib.axes.Axes
    """

    # Convert to numpy arrays for improved efficiency
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    n_classes = len(labels)

    # Perform random sampling if dataset is too large
    if len(y_true) > max_samples:
        print(f"Large dataset detected ({len(y_true)} samples). Sampling {max_samples} for efficiency.")
        indices = np.random.choice(len(y_true), max_samples, replace=False)
        y_true = y_true[indices]
        y_scores = y_scores[indices]

    # Binarize labels for multi-class evaluation
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    # Color mapping for different classes
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    # Store interpolation points for all classes for micro averaging
    all_recall = np.linspace(0, 1, n_thresholds)
    mean_precision = np.zeros_like(all_recall)

    # Plot PR curve for each class
    for i in range(n_classes):
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])

        # Calculate average precision
        avg_precision = average_precision_score(y_true_bin[:, i], y_scores[:, i])

        # Interpolate curve for efficiency improvement
        if len(precision) > n_thresholds:
            # Use numpy interpolation function for higher efficiency
            recall_interp = np.linspace(recall.min(), recall.max(), n_thresholds // 2)
            precision_interp = np.interp(recall_interp, recall[::-1], precision[::-1])
        else:
            recall_interp = recall
            precision_interp = precision

        # Plot PR curve
        ax.plot(recall_interp, precision_interp,
                color=colors[i], linewidth=2, alpha=0.8,
                label=f'{labels[i]} (AP={avg_precision:.3f})')

        # Accumulate for micro average calculation
        mean_precision += np.interp(all_recall, recall_interp, precision_interp)

    # Calculate and plot micro average
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true_bin.ravel(), y_scores.ravel())
    avg_precision_micro = average_precision_score(y_true_bin, y_scores, average='micro')

    # Sample micro average curve as well
    if len(precision_micro) > n_thresholds:
        recall_micro_interp = np.linspace(recall_micro.min(), recall_micro.max(), n_thresholds)
        precision_micro_interp = np.interp(recall_micro_interp,
                                           recall_micro[::-1], precision_micro[::-1])
    else:
        recall_micro_interp = recall_micro
        precision_micro_interp = precision_micro

    ax.plot(recall_micro_interp, precision_micro_interp,
            color='gold', linewidth=3, linestyle='--',
            label=f'Micro-avg (AP={avg_precision_micro:.3f})')

    # Plot macro average
    mean_precision /= n_classes
    macro_ap = mean_precision.mean()
    ax.plot(all_recall, mean_precision,
            color='navy', linewidth=3, linestyle=':',
            label=f'Macro-avg (AP={macro_ap:.3f})')

    # Set chart properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)

    # Modify title to include micro and macro AP information
    full_title = f"{title}\nMicro AP: {avg_precision_micro:.3f} | Macro AP: {macro_ap:.3f}"
    ax.set_title(full_title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add baseline
    baseline = np.sum(y_true_bin, axis=0) / len(y_true_bin)
    ax.axhline(y=baseline.mean(), color='red', linestyle='--', alpha=0.5,
               label=f'Baseline ({baseline.mean():.3f})')

    return ax


def plt_f1_scores(ax: matplotlib.axes.Axes, y_true: np.ndarray, y_scores: np.ndarray,
                  labels: List[str], title: str = 'F1 Scores by Threshold') -> matplotlib.axes.Axes:
    """
    Plot F1 scores as a function of classification thresholds for threshold optimization.

    This function visualizes how F1 scores change across different classification thresholds
    for each class, helping to identify optimal thresholds for multi-class classification.
    It includes individual class F1 curves, global average, and highlights the best 
    threshold points for each class and globally.

    :param ax: The matplotlib axes object for plotting.
    :type ax: matplotlib.axes.Axes
    :param y_true: True labels as class indices.
    :type y_true: np.ndarray
    :param y_scores: Prediction probability scores for each class.
    :type y_scores: np.ndarray
    :param labels: List of class label names for legend display.
    :type labels: List[str]
    :param title: Title for the plot.
    :type title: str
    :return: The modified axes object with the plotted F1 curves.
    :rtype: matplotlib.axes.Axes
    """

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    n_classes = len(labels)

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    thresholds = np.linspace(0, 1, 50)  # Reduce threshold count for efficiency
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    # Store best thresholds and F1 scores for each class
    best_thresholds = []
    best_f1_scores = []

    # Store global F1 scores (average across all classes)
    global_f1_scores = []

    for threshold in thresholds:
        class_f1_scores = []

        for i in range(n_classes):
            y_pred = (y_scores[:, i] >= threshold).astype(int)

            # Fast F1 score calculation
            tp = np.sum((y_pred == 1) & (y_true_bin[:, i] == 1))
            fp = np.sum((y_pred == 1) & (y_true_bin[:, i] == 0))
            fn = np.sum((y_pred == 0) & (y_true_bin[:, i] == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            class_f1_scores.append(f1)

        global_f1_scores.append(np.mean(class_f1_scores))

    # Find global best threshold and F1 score
    global_best_idx = np.argmax(global_f1_scores)
    global_best_threshold = thresholds[global_best_idx]
    global_best_f1 = global_f1_scores[global_best_idx]

    # Plot F1 curve for each class and find best points
    for i in range(n_classes):
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_scores[:, i] >= threshold).astype(int)

            # Fast F1 score calculation
            tp = np.sum((y_pred == 1) & (y_true_bin[:, i] == 1))
            fp = np.sum((y_pred == 1) & (y_true_bin[:, i] == 0))
            fn = np.sum((y_pred == 0) & (y_true_bin[:, i] == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)

        # Find best threshold for current class
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        best_thresholds.append(best_threshold)
        best_f1_scores.append(best_f1)

        # Plot F1 curve
        ax.plot(thresholds, f1_scores, color=colors[i], linewidth=2,
                label=f'{labels[i]} (max F1={best_f1:.3f})')

        # Mark best point
        ax.scatter(best_threshold, best_f1, color=colors[i], s=100,
                   marker='o', edgecolors='black', linewidth=2, zorder=5)

        # Add text annotation for best threshold
        ax.annotate(f'{best_threshold:.2f}',
                    xy=(best_threshold, best_f1),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color=colors[i], fontweight='bold')

    # Plot global average F1 curve
    ax.plot(thresholds, global_f1_scores, color='red', linewidth=3,
            linestyle='--', alpha=0.8, label=f'Global Avg (max F1={global_best_f1:.3f})')

    # Mark global best point
    ax.scatter(global_best_threshold, global_best_f1, color='red', s=150,
               marker='*', edgecolors='black', linewidth=2, zorder=6)

    # Add text annotation for global best threshold
    ax.annotate(f'{global_best_threshold:.2f}',
                xy=(global_best_threshold, global_best_f1),
                xytext=(5, -15), textcoords='offset points',
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)

    # Modify title to include global best threshold and F1 score information
    full_title = f"{title}\nGlobal Best: Threshold={global_best_threshold:.3f}, F1={global_best_f1:.3f}"
    ax.set_title(full_title, fontsize=14, fontweight='bold')

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return ax
