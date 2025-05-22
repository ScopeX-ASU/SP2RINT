# import re
# import torch
# from torch.utils.data import Dataset, DataLoader
# from typing import Tuple, Optional
# import random


# class VowelDataset(Dataset):
#     def __init__(
#         self,
#         root: str,
#         split: str = "train",
#         n_train_speakers: int = 48,
#         train_valid_split_ratio: Tuple[float, float] = (0.9, 0.1),
#         n_input_features: int = 10,
#         n_total_speakers: int = 90,
#         seed: int = 42,
#         n_valid_samples: Optional[int] = None,
#         n_test_samples: Optional[int] = None,
#     ):
#         assert split in {"train", "valid", "test"}
#         self.file_path = root + "/vowel.data"
#         self.split = split
#         self.n_input_features = n_input_features
#         self.n_train_speakers = n_train_speakers
#         self.train_valid_split_ratio = train_valid_split_ratio
#         self.n_total_speakers = n_total_speakers
#         self.seed = seed
#         self.n_valid_samples = n_valid_samples
#         self.n_test_samples = n_test_samples

#         self.max_value = 0
#         self.min_value = 100

#         self.data = self._load_and_split()

#     def _load_and_split(self):
#         with open(self.file_path, "r") as f:
#             raw = f.read()

#         # Extract float vectors
#         matches = re.findall(r'{([^{}]+)}', raw)
#         all_data = []
#         for idx, match in enumerate(matches):
#             values = list(map(float, match.strip().split(',')))
#             self.max_value = max(self.max_value, max(v for v in values))
#             self.min_value = min(self.min_value, min(v for v in values))
#             if len(values) != self.n_input_features:
#                 continue
#             speaker_id = idx // 11  # 11 vowels per speaker
#             vowel_id = idx % 11
#             all_data.append((values, vowel_id, speaker_id))

#         # Separate by speaker group
#         train_valid_data = [item for item in all_data if item[2] < self.n_train_speakers]
#         test_data = [item for item in all_data if item[2] >= self.n_train_speakers]

#         if self.split == "test":
#             test = [(x, y) for (x, y, _) in test_data]
#             return test[:self.n_test_samples] if self.n_test_samples else test

#         # Now split train_valid_data by samples
#         random.seed(self.seed)
#         random.shuffle(train_valid_data)

#         total = len(train_valid_data)
#         n_train = int(self.train_valid_split_ratio[0] * total)
#         train = [(x, y) for (x, y, _) in train_valid_data[:n_train]]
#         valid = [(x, y) for (x, y, _) in train_valid_data[n_train:]]

#         if self.split == "train":
#             return train
#         else:
#             return valid[:self.n_valid_samples] if self.n_valid_samples else valid

#     def __getitem__(self, index):
#         x, y = self.data[index]

#         x = [((v - self.min_value) / (self.max_value - self.min_value)) * torch.pi for v in x]
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

#     def __len__(self):
#         return len(self.data)




# def test_vowel_dataset():
#     root = "./data/connectionist"  # change to the directory containing 'vowel.data'

#     train_dataset = VowelDataset(
#         root=root,
#         split="train",
#         n_train_speakers=72,
#         train_valid_split_ratio=(0.8, 0.2),
#         seed=0,
#     )
#     valid_dataset = VowelDataset(
#         root=root,
#         split="valid",
#         n_train_speakers=72,
#         train_valid_split_ratio=(0.8, 0.2),
#         seed=0,
#     )
#     test_dataset = VowelDataset(
#         root=root,
#         split="test",
#         n_train_speakers=72,
#         seed=0,
#     )

#     print(f"Train samples: {len(train_dataset)}")
#     print(f"Valid samples: {len(valid_dataset)}")
#     print(f"Test samples:  {len(test_dataset)}")

#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     valid_loader = DataLoader(valid_dataset, batch_size=16)
#     test_loader = DataLoader(test_dataset, batch_size=16)

#     for batch in train_loader:
#         x, y = batch
#         print("Train batch:", x.shape, y.shape)
#         print("Sample input:", x[0])
#         print("Sample label:", y[0])
#         break

#     for batch in valid_loader:
#         x, y = batch
#         print("Valid batch:", x.shape, y.shape)
#         break

#     for batch in test_loader:
#         x, y = batch
#         print("Test batch:", x.shape, y.shape)
#         break

# if __name__ == "__main__":
#     test_vowel_dataset()


import re
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import random


class VowelDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        train_valid_split_ratio: Tuple[float, float] = (0.8, 0.2),
        test_split_ratio: float = 0.2,
        n_input_features: int = 10,
        seed: int = 42,
        n_valid_samples: Optional[int] = None,
        n_test_samples: Optional[int] = None,


#         self,
#         root: str,
#         split: str = "train",
#         n_train_speakers: int = 48,
#         train_valid_split_ratio: Tuple[float, float] = (0.9, 0.1),
#         n_input_features: int = 10,
#         n_total_speakers: int = 90,
#         seed: int = 42,
#         n_valid_samples: Optional[int] = None,
#         n_test_samples: Optional[int] = None,
    ):
        assert split in {"train", "valid", "test"}
        self.file_path = root + "/vowel.data"
        self.split = split
        self.n_input_features = n_input_features
        self.train_valid_split_ratio = train_valid_split_ratio
        self.test_split_ratio = test_split_ratio
        self.seed = seed
        self.n_valid_samples = n_valid_samples
        self.n_test_samples = n_test_samples

        self.max_value = 0
        self.min_value = 100

        self.data = self._load_and_split()

    def _load_and_split(self):
        with open(self.file_path, "r") as f:
            raw = f.read()

        # Extract float vectors
        matches = re.findall(r'{([^{}]+)}', raw)
        all_data = []
        for idx, match in enumerate(matches):
            values = list(map(float, match.strip().split(',')))
            if len(values) != self.n_input_features:
                continue
            self.max_value = max(self.max_value, max(values))
            self.min_value = min(self.min_value, min(values))

            speaker_id = idx // 11
            vowel_id = idx % 11
            all_data.append((values, vowel_id, speaker_id))

        # Shuffle all data
        random.seed(self.seed)
        random.shuffle(all_data)

        total = len(all_data)
        n_test = int(total * self.test_split_ratio)
        test_data = all_data[:n_test]
        train_valid_data = all_data[n_test:]

        n_train = int(self.train_valid_split_ratio[0] * len(train_valid_data))
        train_data = train_valid_data[:n_train]
        valid_data = train_valid_data[n_train:]

        if self.split == "train":
            return [(x, y) for (x, y, _) in train_data]
        elif self.split == "valid":
            valid = [(x, y) for (x, y, _) in valid_data]
            return valid[:self.n_valid_samples] if self.n_valid_samples else valid
        else:  # test
            test = [(x, y) for (x, y, _) in test_data]
            return test[:self.n_test_samples] if self.n_test_samples else test

    def __getitem__(self, index):
        x, y = self.data[index]
        x = [((v - self.min_value) / (self.max_value - self.min_value)) * torch.pi for v in x]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.data)

def test_vowel_dataset():
    root = "./data/connectionist"

    train_dataset = VowelDataset(
        root=root,
        split="train",
        seed=0,
    )
    valid_dataset = VowelDataset(
        root=root,
        split="valid",
        seed=0,
    )
    test_dataset = VowelDataset(
        root=root,
        split="test",
        seed=0,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    for batch in train_loader:
        x, y = batch
        print("Train batch:", x.shape, y.shape)
        print("Sample input:", x[0])
        print("Sample label:", y[0])
        break

    for batch in valid_loader:
        x, y = batch
        print("Valid batch:", x.shape, y.shape)
        break

    for batch in test_loader:
        x, y = batch
        print("Test batch:", x.shape, y.shape)
        break

if __name__ == "__main__":
    test_vowel_dataset()
