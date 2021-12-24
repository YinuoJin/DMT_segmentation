import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import accuracy_score, f1_score


##########################################
#  Plots & Transformation
##########################################


def get_contour(img, external=True, draw=False):
    """
    Get contour of image mask or prediction
    Returns
    -------
    contours : list
        List of contour pixel coordinates
    img_processed : np.ndarray
        binarized ndarray with img.shape (1 as contour & 0 as background)
    """

    mode = cv2.RETR_EXTERNAL if external else cv2.RETR_TREE
    img_copy = img.copy()
    img_copy = np.round(img_copy * 255.0).astype(np.uint8)
    _, thresh = cv2.threshold(img_copy, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, mode, cv2.CHAIN_APPROX_NONE)

    if not draw:
        return contours
    else:
        img_processed = np.zeros_like(img)
        cv2.drawContours(img_processed, contours, -1, (255, 255, 255), 1)
        img_processed = (img_processed / 255.0 > 0).astype(np.uint8)  # Convert back from [0, 255] to [0, 1]

    return contours, img_processed


def calc_acc(y_true, y_pred):
    accuracies = []
    for r1, r2 in zip(y_true, y_pred):
        accuracies.append(accuracy_score(r1.flatten(), r2.flatten()))

    return np.mean(accuracies)


def calc_f1(y_true, y_pred):
    accuracies = []
    for r1, r2 in zip(y_true, y_pred):
        accuracies.append(f1_score(r1.flatten(), r2.flatten(), average='weighted'))

    return np.mean(accuracies)


def calc_mse(y_true, y_pred):
    mse_score = np.einsum('bhw -> b', np.power(y_true - y_pred, 2)).mean()
    return mse_score


def calc_hausdorff(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Inconsistent dimensions between prediction and ground-truth matrix"
    hd_list = []
    for r1, r2 in zip(y_true, y_pred):
        # contour transformation & hausdorff dist. calculation
        r1_coords = np.asarray(np.where(r1)).T
        r2_coords = np.asarray(np.where(r2)).T

        hd = max(
            directed_hausdorff(r1_coords, r2_coords)[0],
            directed_hausdorff(r2_coords, r1_coords)[0]
        )
        hd_list.append(hd)

    return np.mean(hd_list)


def plot_img_3d_distribution(img, figsize=(8, 6)):
    """
    Plot 3D value distribution of the given image
    """
    height, width = img.shape

    img_vals = np.zeros((height * width, 3))
    img_vals[:, 0] = np.repeat(np.arange(height), width)
    img_vals[:, 1] = np.tile(np.arange(width), height)
    img_vals[:, 2] = img.flatten()

    df = pd.DataFrame(img_vals, columns=['X', 'Y', 'Z'])

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    ax.view_init(45, 45)
    plt.show()
    plt.close()


def plot_img_histogram(img, figsize=(8, 6)):
    """Image histogram visualization, assume image has shape [C, H, W]"""
    # reference: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
    img = img.transpose((1, 2, 0))  # change image order to [H, W, C]
    hist, bins = np.histogram(img.flatten(), 256, [0, 1])

    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    plt.figure(figsize=figsize)
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 1])
    plt.xlim([0, 1])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    plt.close()


##########################################
#  Loss functions
##########################################

class ShapeBCELoss(nn.Module):
    """
    Shape-aware BCE loss based on distance / shape-based loss functions
    """
    def __init__(self):
        super(ShapeBCELoss, self).__init__()

    def forward(self, y_true, y_pred, weight):
        return F.binary_cross_entropy_with_logits(y_pred, y_true, weight)


class DMTLoss(nn.Module):
    """
    Discrete-Morse-Complex loss
    """
    def __init__(self):
        super(DMTLoss, self).__init__()

    def forward(self, y_true, y_pred, ms):
        y1 = torch.mul(y_pred, ms)
        y2 = torch.mul(y_true, ms)
        return F.binary_cross_entropy_with_logits(y1, y2)
