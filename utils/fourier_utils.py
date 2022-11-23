import pyfftw
import numpy as np


def fft(image, fftw_object=False):

    if fftw_object:
        ft_image = fftw_object(image)
    else:
        ft_image = np.fft.rfft2(image)

    return ft_image

def ifft(ft_image, ifftw_object=False):

    if ifftw_object:
        image = ifftw_object(ft_image)
    else:
        image = np.fft.irfft2(ft_image)

    return image


def get_pyfftw_object(
                        input_shape,
                        input_dtype='float64',
                        output_dtype='complex128',
                        input_shape_is_real=True,
                        fast_search=False
                        ):
    """Get an pyFFTw object for very fast FFTs

    Note1: by default pyFFTw does real FFTs - therefore output shape is, for
    an input volume with dimensions [X, Y, Z] is automatically set to
    [X, Y, (Z / 2) + 1]
    Note2: This is for 3D FFTs
    """

    if input_shape_is_real:
        output_shape = [
                        input_shape[0],
                        int((input_shape[1] / 2) + 1)
                        ]
    else:
        output_shape = input_shape

    input = pyfftw.empty_aligned(input_shape, dtype=input_dtype)
    output_shape = [
                    input_shape[0],
                    int((input_shape[1] / 2) + 1)
                    ]
    output = pyfftw.empty_aligned(output_shape, dtype=output_dtype)

    if fast_search:
        fftw_object = pyfftw.FFTW(
                                input,
                                output,
                                axes=(0,1),
                                flags=('FFTW_ESTIMATE',),)
    else:
        fftw_object = pyfftw.FFTW(input, output, axes=(0,1))

    return fftw_object


def get_pyifftw_object(
                        input_shape,
                        input_dtype='complex128',
                        output_dtype='float64',
                        input_shape_is_real=False,
                        fast_search=False
                        ):
    """Get an pyFFTw object for very fast iFFTs

    NOTE: NEED TO CHECK INPUT/OUTPUT DTYPES

    Note1: by default pyFFTw does real FFTs - therefore output shape is, for
    an input volume with dimensions [X, Y, Z] is automatically set to
    [X, Y, (Z / 2) + 1]
    Note2: This is for 3D FFTs
    """
    #output = pyfftw.empty_aligned(output_shape, dtype=output_dtype)

    if input_shape_is_real:
        output_shape = input_shape
        input_shape = [
                        input_shape[0],
                        int((input_shape[1] / 2) + 1)
                        ]
    else:
        input_shape = input_shape
        output_shape = [
                        input_shape[0],
                        int((input_shape[1] - 1) * 2)
                        ]

    input = pyfftw.empty_aligned(input_shape, dtype=input_dtype)
    output = pyfftw.empty_aligned(output_shape, dtype=output_dtype)

    if fast_search:
        ifftw_object = pyfftw.FFTW(
                                input,
                                output,
                                axes=(0,1),
                                direction='FFTW_BACKWARD',
                                flags=('FFTW_ESTIMATE',),)
    else:
        ifftw_object = pyfftw.FFTW(
                                input,
                                output,
                                axes=(0,1),
                                direction='FFTW_BACKWARD',)

    return ifftw_object


def get_good_box_for_fft(input_dimension):
    """Helper function to find convenient box size for fourier transforms
    """
    power_two = 2
    power_three = 3
    power_five = 5
    power_seven = 7
    box_size = 0

    while True:
        if input_dimension <= power_two:
            new_dimension = power_two
            break
        elif input_dimension <= power_three:
            new_dimension = power_three
            break
        elif input_dimension <= power_five:
            new_dimension = power_five
            break
        elif input_dimension <= power_seven:
            new_dimension = power_seven
            break
        else:
            power_two = power_two * 2
            power_three = power_three * 2
            power_five = power_five * 2
            power_seven = power_seven * 2

    if new_dimension * 0.75 > input_dimension:
        new_dimension = input_dimension

    return new_dimension


def get_labelled_shells_2d(output_shape):
    """
    Get labelled fourier shells for a nD volume
    """

    x_shape = output_shape[1]
    y_shape = output_shape[0]

    # get the fourier frequencies in each dimension as a 3d grid
    qx_ = np.fft.rfftfreq(x_shape)#*x_shape
    qy_ = np.fft.fftfreq(y_shape)#*y_shape
    qx, qy, = np.meshgrid(qy_, qx_, indexing='ij')

    # use the radius to calculate which fourier shell a given pixel is in
    qx_max = qx.max()
    qr = np.sqrt(qx**2+qy**2)
    qmax = np.max(qr)
    qstep = np.min(qx_[qx_>0])
    nbins = int(qmax/qstep)
    qbins = np.linspace(0,nbins*qstep,nbins+1)

    # label the fourier shells from 0 -> n-1, where n is the no of shells
    labelled_shells = np.searchsorted(qbins, qr, "right")
    labelled_shells -= 1

    return labelled_shells


def get_labelled_shells_3d(output_shape):
    """
    Get labelled fourier shells for a nD volume
    """

    x_shape = output_shape[2]
    y_shape = output_shape[1]
    z_shape = output_shape[0]

    # get the fourier frequencies in each dimension as a 3d grid
    qx_ = np.fft.rfftfreq(x_shape)#*x_shape
    qy_ = np.fft.fftfreq(y_shape)#*y_shape
    qz_ = np.fft.fftfreq(z_shape)#*z_shape
    qx, qy, qz = np.meshgrid(qz_,qy_,qx_,indexing='ij')

    # use the radius to calculate which fourier shell a given pixel is in
    qx_max = qx.max()
    qr = np.sqrt(qx**2+qy**2+qz**2)
    qmax = np.max(qr)
    qstep = np.min(qx_[qx_>0])
    nbins = int(qmax/qstep)
    qbins = np.linspace(0,nbins*qstep,nbins+1)

    # label the fourier shells from 0 -> n-1, where n is the no of shells
    labelled_shells = np.searchsorted(qbins, qr, "right")
    labelled_shells -= 1

    return labelled_shells


def get_labelled_shells(output_shape):

    if len(output_shape) == 2:
        labelled_shells = get_labelled_shells_2d(output_shape)
    elif len(output_shape) == 3:
        labelled_shells = get_labelled_shells_3d(output_shape)
    else:
        raise ValueError("hcmv_pick can only handle 2D or 3D inputs for fourier"
            "transform computations")

    return labelled_shells
