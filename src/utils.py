from PIL import Image
import numpy as np

def read_image(path, dtype=np.float32, color=True, convert_np=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behavior.
            If :obj:`False`, this function returns a grayscale image.
        convert_np (bool): This option determines wether convert PIL's Image to numpy ndarray.
            If :obj:`True`,  return image in numpy ndarray.
            If :obj:`False`, this function returns PIL's Image .
    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
    finally:
        if hasattr(f, 'close'):
            f.close()
    
    if not convert_np:
        return img
    
    img = np.asarray(img, dtype=dtype)
    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))
