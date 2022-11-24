
import numpy as np

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
    filt_im = np.fft.irfft2(ft_im, axes=(0,1))
    im[:, :, 0] = filt_im
    im[:, :, 1] = filt_im
    im[:, :, 2] = filt_im
    return im


def enhance_edge_features(image):

    kernel = np.array([  0, -1,  0,
                        -1,  5, -1,
                         0, -1,  0,])

    filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    return filtered_image
