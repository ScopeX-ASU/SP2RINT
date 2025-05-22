import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

__all__ = ["QuickDrawDataset"]


resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


class QuickDrawDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        train_valid_split_ratio: List[float] = (0.9, 0.1),
        center_crop: bool = True,
        resize: bool = True,
        resize_mode: str = "bicubic",
        binarize: bool = False,
        binarize_threshold: float = 0.5,
        classes_of_interest: List[str] = None,
        num_classes: Optional[int] = 10,
        n_samples_per_class: int = None,
        n_bytes_per_class: int = None,
    ):
        self.root = root
        self.split = split
        self.train_valid_split_ratio = train_valid_split_ratio
        self.center_crop = center_crop
        self.resize = resize
        self.resize_mode = resize_modes[resize_mode]
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.classes_of_interest = classes_of_interest
        self.num_classes = num_classes
        self.n_samples_per_class = n_samples_per_class
        self.n_bytes_per_class = n_bytes_per_class

        self.data, self.labels = self.load_data()
        self.n_instance = len(self.data)

    def load_data(self):
        data = []
        labels = []

        tran = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        if self.center_crop:
            tran.append(transforms.CenterCrop(28))
        if self.resize:
            tran.append(transforms.Resize(28, interpolation=self.resize_mode))
        transform = transforms.Compose(tran)

        if self.classes_of_interest:
            classes = self.classes_of_interest
        else:
            all_classes = [f[:-4] for f in os.listdir(self.root) if f.endswith(".npy")]
            if self.num_classes:
                classes = all_classes[: self.num_classes]
            else:
                classes = all_classes

        for idx, class_name in enumerate(classes):
            class_path = os.path.join(self.root, f"{class_name}.npy")
            class_data = np.load(class_path)

            # print(class_data.shape)
            # exit(0)
            if self.split == "train" or self.split == "valid":
                if self.n_samples_per_class is not None:
                    class_data = class_data[: self.n_samples_per_class]
                elif self.n_bytes_per_class is not None:
                    bytes_per_sample = class_data.nbytes // len(class_data)
                    n_samples = self.n_bytes_per_class // bytes_per_sample
                    class_data = class_data[:n_samples]

                for img in class_data:
                    img = img.reshape(28, 28)
                    img = transform(img)
                    if self.binarize:
                        img = (img > self.binarize_threshold).float()
                    data.append(img)
                    labels.append(idx)
            else:
                if self.n_samples_per_class is not None:
                    class_data = class_data[
                        self.n_samples_per_class : self.n_samples_per_class + 1000
                    ]
                elif self.n_bytes_per_class is not None:
                    bytes_per_sample = class_data.nbytes // len(class_data)
                    n_samples = self.n_bytes_per_class // bytes_per_sample
                    class_data = class_data[n_samples : n_samples + 1000]

                for img in class_data:
                    img = img.reshape(28, 28)
                    img = transform(img)
                    if self.binarize:
                        img = (img > self.binarize_threshold).float()
                    data.append(img)
                    labels.append(idx)

        # print
        data = torch.stack(data)
        labels = torch.tensor(labels)

        if self.split == "train" or self.split == "valid":
            train_len = int(self.train_valid_split_ratio[0] * len(data))
            valid_len = len(data) - train_len
            train_data, valid_data = torch.utils.data.random_split(
                list(zip(data, labels)),
                [train_len, valid_len],
                generator=torch.Generator().manual_seed(1),
            )

            if self.split == "train":
                data, labels = zip(*train_data)
            else:
                data, labels = zip(*valid_data)

            data = torch.stack(data)
            labels = torch.tensor(labels)

        # print(data.shape, labels.shape)
        # exit(0)

        return data, labels

    def __getitem__(self, index: int):
        img = self.data[index]
        label = self.labels[index]
        return img, label

    def __len__(self) -> int:
        return self.n_instance


def test():
    quick_draw = QuickDrawDataset(
        root="/home/dataset/quickdraw",
        split="test",
        train_valid_split_ratio=[0.9, 0.1],
        center_crop=True,
        resize=True,
        resize_mode="bilinear",
        binarize=False,
        binarize_threshold=0.5,
        classes_of_interest=None,
        num_classes=10,
        n_samples_per_class=6000,
        n_bytes_per_class=None,
    )
    data, label = quick_draw.__getitem__(2000)
    print(data.size(), label.size())
    print(label)
    print("finish")


if __name__ == "__main__":
    test()


# if __name__ == "__main__":
#     test()
