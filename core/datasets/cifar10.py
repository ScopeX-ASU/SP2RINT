from typing import List

import torch
from pyutils.general import logger
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

__all__ = ["CIFAR10Dataset"]


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        train_valid_split_ratio: List[float] = (0.9, 0.1),
        center_crop: bool = True,
        resize: bool = True,
        resize_mode: str = "bicubic",
        binarize: bool = False,
        binarize_threshold: float = 0.1307,
        grayscale: bool = False,
        digits_of_interest: List[float] = list(range(10)),
        n_test_samples: int = 10000,
        n_valid_samples: int = 5000,
        fashion: bool = False,
        augment: bool = False,
    ):
        self.root = root
        self.split = split
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.center_crop = center_crop
        self.resize = resize
        self.resize_mode = resize_modes[resize_mode]
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.grayscale = grayscale
        self.digits_of_interest = digits_of_interest
        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples
        self.fashion = fashion
        self.augment = augment

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        if self.augment:
            transform_train = [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(self.resize, interpolation=self.resize_mode),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform_train = []
            if not self.center_crop == 32:
                transform_train.append(transforms.CenterCrop(self.center_crop))
            if not self.resize == 32:
                transform_train.append(
                    transforms.Resize(self.resize, interpolation=self.resize_mode)
                )

        transform_test = []
        if not self.center_crop == 32:
            transform_test.append(transforms.CenterCrop(self.center_crop))
        if not self.resize == 32:
            transform_test.append(
                transforms.Resize(self.resize, interpolation=self.resize_mode)
            )

        if self.grayscale:  # TODO
            transform_train += [
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize(
                    (0.2989 * 0.4914 + 0.587 * 0.4822 + 0.114 * 0.4465,),
                    (
                        (
                            (0.2989 * 0.2023) ** 2
                            + (0.587 * 0.1994) ** 2
                            + (0.114 * 0.2010) ** 2
                        )
                        ** 0.5,
                    ),
                ),
            ]
            transform_test += [
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize(
                    (0.2989 * 0.4914 + 0.587 * 0.4822 + 0.114 * 0.4465,),
                    (
                        (
                            (0.2989 * 0.2023) ** 2
                            + (0.587 * 0.1994) ** 2
                            + (0.114 * 0.2010) ** 2
                        )
                        ** 0.5,
                    ),
                ),
            ]
        else:
            transform_train += [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
            transform_test += [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]

        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)

        if self.split == "train" or self.split == "valid":
            train_valid = datasets.CIFAR10(
                self.root,
                train=True,
                download=True,
                transform=transform_train if self.split == "train" else transform_test,
            )
            targets = torch.tensor(train_valid.targets)
            idx, _ = torch.stack(
                [targets == number for number in self.digits_of_interest]
            ).max(dim=0)
            # targets = targets[idx]
            train_valid.targets = targets[idx].numpy().tolist()
            train_valid.data = train_valid.data[idx]

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            split = [train_len, len(train_valid) - train_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )
            if self.split == "train":
                self.data = train_subset
            else:
                if self.n_valid_samples is None:
                    # use all samples in valid set
                    self.data = valid_subset
                else:
                    # use a subset of valid set, useful to speedup evo search
                    valid_subset.indices = valid_subset.indices[: self.n_valid_samples]
                    self.data = valid_subset
                    logger.warning(
                        f"Only use the front "
                        f"{self.n_valid_samples} images as "
                        f"VALID set."
                    )

        else:
            test = datasets.CIFAR10(self.root, train=False, transform=transform_test)
            targets = torch.tensor(test.targets)
            idx, _ = torch.stack(
                [targets == number for number in self.digits_of_interest]
            ).max(dim=0)
            test.targets = targets[idx].numpy().tolist()
            test.data = test.data[idx]
            if self.n_test_samples is None:
                # use all samples as test set
                self.data = test
            else:
                # use a subset as test set
                test.targets = test.targets[: self.n_test_samples]
                test.data = test.data[: self.n_test_samples]
                self.data = test
                logger.warning(
                    f"Only use the front {self.n_test_samples} " f"images as TEST set."
                )

    def __getitem__(self, index: int):
        img = self.data[index][0]
        if self.binarize:
            img = 1.0 * (img > self.binarize_threshold) + -1.0 * (
                img <= self.binarize_threshold
            )

        digit = self.digits_of_interest.index(self.data[index][1])
        return img, torch.tensor(digit).long()
        # instance = {'image': img, 'digit': digit}
        # return instance

    def __len__(self) -> int:
        return self.n_instance


def test():
    cifar10 = CIFAR10Dataset(
        root="/home/dataset/cifar10",
        split="train",
        train_valid_split_ratio=[0.9, 0.1],
        center_crop=32,
        resize=32,
        resize_mode="bilinear",
        binarize=False,
        binarize_threshold=0.1307,
        grayscale=True,
        digits_of_interest=(3, 6),
        n_test_samples=100,
        n_valid_samples=1000,
        fashion=True,
    )
    data, labels = cifar10.__getitem__(20)
    print(data.size(), labels.size())
    print("finish")


if __name__ == "__main__":
    test()
