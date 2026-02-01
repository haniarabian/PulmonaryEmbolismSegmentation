import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.datasets.folder import default_loader


def _make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            item = path
            im = Image.open(item)
            im = np.asanyarray(im)
            im = im.sum()
            if im > 0:
                images.append(item)
    return images


class MSDataset(data.Dataset):
    def __init__(
        self,
        root,
        input_dim=256,
        mean=0.3511,
        std=0.2331,
        transform=None,
        loader=default_loader,
        seq_size=3,
        sliding_window=True,
    ):
        self.root = root
        self.input_dim = input_dim
        self.transform = transform
        self.loader = loader
        self.mean = mean
        self.std = std
        self.seq_size = seq_size
        self.sliding_window = sliding_window
        self.imgs = _make_dataset(root)

    def __len__(self):
        if self.sliding_window:
            return len(self.imgs) - self.seq_size + 1
        return len(self.imgs) // self.seq_size

    def __getitem__(self, index):
        imgs = torch.Tensor(self.seq_size, 3, self.input_dim, self.input_dim)
        pil_imgs = []

        for i in range(self.seq_size):
            if self.sliding_window == False:
                path = self.imgs[(index) * self.seq_size + i]
            else:
                path = self.imgs[index + i]
            img = self.loader(path)
            pil_imgs.append(img)

        for i in range(self.seq_size):
            img = pil_imgs[i]
            if self.transform is not None:  # trasformazioni immagini
                img = self.transform(img)
            m1 = img.min()
            m2 = img.max()
            if m1 != m2:
                img = (img - m1) / (m2 - m1)
                img = (img - self.mean) / self.std
            imgs[i] = img
        return path, imgs
