import numbers
import numpy as np
import random
import torch

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0,padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        h, w = img.shape[1], img.shape[2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        device = img.device
        dtype = img.dtype
        img = img.detach().cpu().numpy()
        if self.padding is not None:
            img = np.transpose(img, (1, 2, 0))
            temp = np.zeros((img.shape[0]+2*self.padding, img.shape[1]+2*self.padding, img.shape[2]))
            # img = np.pad(img, self.padding, self.padding_mode, constant_values=self.fill)
            temp[self.padding: self.padding+img.shape[0], self.padding: self.padding+img.shape[1], :] = img
            img = temp
            img = np.transpose(img, (2, 0, 1))
        i, j, h, w = self.get_params(img, self.size)
        img =  img[:, i:i+h, j:j+w]
        img = torch.Tensor(img).to(dtype=dtype, device=device)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)
