from typing import List
import os
import scipy.io as sio
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from pyutils.general import logger, print_stat
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

__all__ = ["SVHNDataset"]

class SVHNDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        train_valid_split_ratio: List[float] = (0.9, 0.1),
        center_crop: int = 32,
        resize: int = 32,
        resize_mode: str = "bicubic",
        binarize: bool = False,
        binarize_threshold: float = 0.1307,
        grayscale: bool = False,
        digits_of_interest: List[int] = list(range(10)),
        n_test_samples: int = 10000,
        n_valid_samples: int = 5000,
        augment: bool = False,
    ):
        self.root = root
        self.split = split
        self.train_valid_split_ratio = train_valid_split_ratio
        self.center_crop = center_crop
        self.resize = resize
        self.resize_mode = resize_modes[resize_mode]
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.grayscale = grayscale
        self.digits_of_interest = digits_of_interest
        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples
        self.augment = augment

        self.train_min = -2.4
        self.train_max = 2.84

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        if self.split in ["train", "valid"]:
            mat_data = sio.loadmat(os.path.join(self.root, "train_32x32.mat"))
        else:
            mat_data = sio.loadmat(os.path.join(self.root, "test_32x32.mat"))

        # Load and preprocess
        images = torch.from_numpy(mat_data["X"]).permute(3, 2, 0, 1).float() / 255.0  # [N, C, H, W]
        labels = torch.from_numpy(mat_data["y"].squeeze().astype("int64"))
        labels[labels == 10] = 0  # label 10 -> 0

        # Filter digits
        mask = torch.isin(labels, torch.tensor(self.digits_of_interest))
        images = images[mask]
        labels = labels[mask]

        # Build transformation
        if self.augment and self.split == "train":
            transform = [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(32, interpolation=self.resize_mode),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform = []
            if self.center_crop != 32:
                transform.append(transforms.CenterCrop(self.center_crop))
            if self.resize != 32:
                transform.append(transforms.Resize(self.resize, interpolation=self.resize_mode))

        if self.grayscale:
            transform += [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        else:
            transform += [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
                transforms.Lambda(lambda x: (x - self.train_min) / (self.train_max - self.train_min) * torch.pi),
            ]

        self.transform = transforms.Compose(transform)

        # Construct dataset
        dataset = [(images[i], labels[i]) for i in range(len(images))]

        if self.split in ["train", "valid"]:
            train_len = int(self.train_valid_split_ratio[0] * len(dataset))
            split = [train_len, len(dataset) - train_len]
            train_set, valid_set = torch.utils.data.random_split(
                dataset, split, generator=torch.Generator().manual_seed(1)
            )
            if self.split == "train":
                self.data = train_set
            else:
                if self.n_valid_samples:
                    valid_set.indices = valid_set.indices[:self.n_valid_samples]
                    logger.warning(
                        f"Only use the front "
                        f"{self.n_valid_samples} images as "
                        f"VALID set."
                    )
                self.data = valid_set
        else:
            if self.n_test_samples:
                dataset = dataset[:self.n_test_samples]
                logger.warning(
                    f"Only use the front {self.n_test_samples} " f"images as TEST set."
                )
            self.data = dataset

    def __getitem__(self, index: int):
        img, label = self.data[index]
        img = self.transform(img)
        if self.binarize:
            img = 1.0 * (img > self.binarize_threshold) + -1.0 * (img <= self.binarize_threshold)
        digit = self.digits_of_interest.index(label.item())
        return img, torch.tensor(digit).long()

    def __len__(self):
        return self.n_instance

if __name__ == "__main__":
    # Example usage
    svhn = SVHNDataset(
        root="./data/svhn",
        split="train",
        grayscale=False,
        digits_of_interest=[0, 1, 2],
        n_test_samples=None,
        n_valid_samples=None,
        center_crop=32,
        resize=32,
    )

    loader = DataLoader(svhn, batch_size=512, shuffle=True)

    global_min = float('inf')
    global_max = float('-inf')

    for imgs, _ in loader:
        batch_min = imgs.min()
        batch_max = imgs.max()
        global_min = min(global_min, batch_min.item())
        global_max = max(global_max, batch_max.item())

    print("Global min:", global_min)
    print("Global max:", global_max)
    # global min: -2.4
    # global max: 2.84