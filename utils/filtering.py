import time

import numpy as np
from scipy import stats

from utils import fourier_utils as ft_utils
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, check_dataset, check_requirements,
                           check_yaml, clean_str, cv2, is_colab, is_kaggle, segments2boxes, unzip_file, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)


def get_relion_style_lowpass_filter(
                                    input_shape,
                                    lowpass_freq,
                                    pixel_size,
                                    cos_edge=2,
                                    ):
    """
    """
    output_shape = list(input_shape)
    cos_edge += 2

    labelled_shells = ft_utils.get_labelled_shells(output_shape)
    f = np.copy(labelled_shells.astype(np.float64))

    # get nearest shell number for cutoff frequency
    # assumes equal pixel size in x, y and z (if 3d)
    shell_cutoff = int(round(
            ((output_shape[0] * pixel_size) / lowpass_freq ), 0))

    # get integer for half the cosine width
    half_cos_width = int(cos_edge / 2)

    # define the limits
    highest_shell = shell_cutoff + half_cos_width
    lowest_shell = shell_cutoff - half_cos_width
    delta = highest_shell - lowest_shell

    # modify labelled_shells to generate the filter
    f[labelled_shells <= lowest_shell] = 1.
    f[labelled_shells >= highest_shell] = 0.
    # make a nice cosine edge
    for n in range(lowest_shell, highest_shell, 1):
        x = (n - lowest_shell) / delta
        x = np.cos(np.pi * x)
        x = (x + 1) / 2
        f[labelled_shells == n] = x

    return f


def apply_fourier_filter(im, fourier_filter):
    ft_im = np.fft.rfft2(im[:, :, 0], axes=(0,1))
    ft_im = ft_im * fourier_filter
    im[:, :, 0] = np.fft.irfft2(ft_im, axes=(0,1))
    im[:, :, 1] = im[:, :, 0]
    im[:, :, 2] = im[:, :, 0]
    return im


def enhance_edge_features(image):

    kernel = np.array([  0, -1,  0,
                        -1,  5, -1,
                         0, -1,  0,])

    filtered_image = np.zeros(image.shape)
    filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel, output=filtered_image)

    return filtered_image

def normalise_to_8bit_range(
                            image,
                            use_central_portion=True,
                            ignore_zeros=True
                            ):
    norm_image = image[:]

    if use_central_portion:
        norm_mean, norm_std, norm_min, norm_max \
                    = _get_mean_of_central_image_portion(
                                                    norm_image,
                                                    ignore_zeros=ignore_zeros
                                                    )
    elif ignore_zeros:
        norm_mean = norm_image[norm_image != 0.].mean()
        norm_min = norm_image[norm_image != 0.].min()
        norm_max = norm_image[norm_image != 0.].max()
        norm_std = norm_image[norm_image != 0.].std()

    else:
        norm_mean = norm_image.mean()
        norm_std = norm_image.std()
        norm_max = norm_image.max()
        norm_min = norm_image.min()

    # normalise
    #norm_image = (norm_image - norm_min) / (norm_max - norm_min)
    norm_image = (norm_image - norm_mean) / norm_std
    m, sig = stats.norm.fit(norm_image)
    norm_image = norm_image * (64 / sig)
    norm_image = norm_image + (128 - stats.norm.fit(norm_image)[0])
    norm_image = np.clip(norm_image, 0, 255)

    return norm_image

def _get_mean_of_central_image_portion(
                                    norm_image,
                                    portion=0.5,
                                    ignore_zeros=True
                                    ):
    frac = (1 / portion) * 2
    qr = int(norm_image.shape[0] / frac)
    qc = int(norm_image.shape[1] / frac)
    norm_box = norm_image[
                    norm_image.shape[0] - qr : norm_image.shape[0] + qr,
                    norm_image.shape[1] - qc : norm_image.shape[1] + qc]
    if ignore_zeros:
        norm_mean, norm_std = stats.norm.fit(norm_box[norm_box != 0.])
        #norm_mean = norm_box[norm_box != 0.].mean()
        #norm_std = norm_box[norm_box != 0.].std()
        norm_min = norm_box[norm_box != 0.].min()
        norm_max = norm_box[norm_box != 0.].max()
    else:
        norm_mean, norm_std = stats.norm.fit(norm_box)
        #norm_mean = norm_box.mean()
        #norm_std = norm_box.std()
        norm_min = norm_box.min()
        norm_max = norm_box.max()

    return norm_mean, norm_std, norm_min, norm_max
