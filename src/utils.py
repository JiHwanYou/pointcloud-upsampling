import numpy as np
import torch
from torchvision import transforms


def rae2xyz(rae: np.ndarray) -> np.ndarray:
    """Convert from rav(range/azimuth/elevation) coordinate to xyz(x/y/z) coordinate."""

    r = rae[:, 0]
    a = np.deg2rad(rae[:, 1])
    e = np.deg2rad(rae[:, 2])

    x = r * np.cos(e) * np.cos(a)
    y = r * np.cos(e) * np.sin(a)
    z = r * np.sin(e)

    return np.stack((x, y, z), axis=1)


def xyz2rae(xyz: np.ndarray) -> np.ndarray:
    """Convert from xyz(x/y/z) coordinate to rae(range/azimuth/elevation) coordinate."""

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    dist_xy = np.sqrt(x**2 + y**2)

    a = np.rad2deg(np.arctan2(y, x))
    e = np.rad2deg(np.arctan2(z, dist_xy))
    r = np.sqrt(x**2 + y**2 + z**2)

    return np.stack((r, a, e), axis=1)


def ndarray2tensor(ndarray: np.ndarray, batch_size=1):
    """Convert ndaaray[H,W,C] to tensor[B,C,H,W].
    Assume batch_size=1.
    """

    row, col, channel = map(int, ndarray.shape)

    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(ndarray)

    if batch_size == None:
        return tensor
    else:
        return torch.reshape(tensor, (batch_size, channel, row, col))


def tensor2ndarray(tensor: torch.Tensor):
    """Convert tensor[B,C,H,W] to ndarray[H,W,C].
    The input tensor batch size must be 1.
    """

    batch_size, channel, row, col = map(int, tensor.shape)

    tensor = tensor.reshape(channel, row, col)
    ndarray = tensor.detach().numpy()

    # Convert from (C, H, W) to (H, W, C)
    ndarray = ndarray.transpose((1, 2, 0))
    return ndarray


