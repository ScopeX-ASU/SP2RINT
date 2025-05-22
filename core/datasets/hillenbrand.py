import os
import sys
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from scipy.io import wavfile
from scipy.fftpack import dct as scipy_dct

__all__ = ["HillenbrandDataset"]

class HillenbrandDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_valid_speakers: int = 10,
        n_test_speakers: int = 10,
        feature_type: str = "padded_signal",  # "mfcc" or "padded_signal"
        random_seed: int = 42,
    ):
        assert split in ["train", "valid", "test"]
        assert feature_type in ["mfcc", "padded_signal"]
        self.root = root
        self.split = split
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_valid_speakers = n_valid_speakers
        self.n_test_speakers = n_test_speakers
        self.feature_type = feature_type
        self.random_seed = random_seed

        self.data = []
        self._prepare_dataset()

    def _extract_features(self, filepath):
        sr, signal = wavfile.read(filepath)
        if signal.ndim > 1:
            signal = signal[:, 0]
        signal = torch.tensor(signal, dtype=torch.float32)

        if self.feature_type == "padded_signal":
            pad_len = self.max_signal_len - signal.shape[0]
            left_pad = pad_len // 2
            right_pad = pad_len - left_pad
            padded_signal = torch.nn.functional.pad(signal, (left_pad, right_pad))
            return padded_signal

        # MFCC feature extraction
        emphasized = torch.cat([signal[:1], signal[1:] - 0.97 * signal[:-1]])
        frame_size = 0.025
        frame_stride = 0.01
        frame_len = int(round(frame_size * sr))
        frame_step = int(round(frame_stride * sr))
        signal_len = emphasized.size(0)
        num_frames = int(torch.ceil(torch.tensor((signal_len - frame_len) / frame_step)).item()) + 1

        pad_signal_len = num_frames * frame_step + frame_len
        z = torch.zeros(pad_signal_len - signal_len)
        pad_signal = torch.cat((emphasized, z))

        indices = (
            torch.arange(0, frame_len).repeat(num_frames, 1) +
            torch.arange(0, num_frames * frame_step, frame_step).unsqueeze(1)
        )
        frames = pad_signal[indices]
        frames *= torch.hamming_window(frame_len)

        NFFT = 512
        mag_frames = torch.abs(torch.fft.rfft(frames, n=NFFT))
        pow_frames = (1.0 / NFFT) * (mag_frames ** 2)

        nfilt = 26
        low_mel = 0.0
        high_mel = 2595.0 * torch.log10(torch.tensor(1.0 + (sr / 2) / 700.0))
        mel_points = torch.linspace(low_mel, high_mel, nfilt + 2)
        hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1)
        bin = torch.floor((NFFT + 1) * hz_points / sr).long()

        fbank = torch.zeros(nfilt, NFFT // 2 + 1)
        for m in range(1, nfilt + 1):
            f_m_minus = bin[m - 1]
            f_m = bin[m]
            f_m_plus = bin[m + 1]

            if f_m_minus == f_m or f_m == f_m_plus:
                continue

            fbank[m - 1, f_m_minus:f_m] = (
                torch.arange(f_m_minus, f_m, dtype=torch.float32) - f_m_minus
            ) / (f_m - f_m_minus)
            fbank[m - 1, f_m:f_m_plus] = (
                f_m_plus - torch.arange(f_m, f_m_plus, dtype=torch.float32)
            ) / (f_m_plus - f_m)

        filter_banks = torch.matmul(pow_frames, fbank.T)
        filter_banks = torch.clamp(filter_banks, min=1e-10)
        filter_banks = 20 * torch.log10(filter_banks)

        mfcc = torch.from_numpy(scipy_dct(filter_banks.numpy(), type=2, axis=1, norm='ortho')[:, :self.n_mfcc])
        mfcc_mean = torch.mean(mfcc, dim=0)
        mfcc_std = torch.std(mfcc, dim=0)

        return torch.cat([mfcc_mean, mfcc_std])

    def _prepare_dataset(self):
        files = [f for f in os.listdir(self.root) if f.endswith(".wav")]
        speaker_dict = defaultdict(list)
        signal_lengths = []

        for fname in files:
            gender = fname[0]
            speaker_id = fname[1:3]
            label = fname[3:5]
            speaker_key = f"{gender}{speaker_id}"
            full_path = os.path.join(self.root, fname)
            speaker_dict[speaker_key].append((full_path, label))

            if self.feature_type == "padded_signal":
                sr, signal = wavfile.read(full_path)
                signal_lengths.append(signal.shape[0])

        if self.feature_type == "padded_signal":
            self.max_signal_len = max(signal_lengths)

        male_speakers = sorted([k for k in speaker_dict if k.startswith("m")])
        female_speakers = sorted([k for k in speaker_dict if k.startswith("w")])

        def split_speakers(speakers):
            np.random.seed(self.random_seed)
            np.random.shuffle(speakers)
            valid = speakers[:self.n_valid_speakers // 2]
            test = speakers[self.n_valid_speakers // 2:self.n_valid_speakers // 2 + self.n_test_speakers // 2]
            train = speakers[self.n_valid_speakers // 2 + self.n_test_speakers // 2:]
            return train, valid, test

        male_train, male_valid, male_test = split_speakers(male_speakers)
        female_train, female_valid, female_test = split_speakers(female_speakers)

        split_map = {
            "train": male_train + female_train,
            "valid": male_valid + female_valid,
            "test": male_test + female_test,
        }
        selected_speakers = set(split_map[self.split])

        for speaker in selected_speakers:
            for filepath, label in speaker_dict[speaker]:
                feature = self._extract_features(filepath)
                self.data.append((feature, label))

        self.label2idx = {label: idx for idx, label in enumerate(sorted(set(l for _, l in self.data)))}
        self.data = [(x, self.label2idx[y]) for x, y in self.data]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def test():
    dataset = HillenbrandDataset(root="../../data/vowels", split="train", feature_type="padded_signal")
    x, y = dataset[0]
    print(x.shape, y)
    print("类别标签及索引映射：", dataset.label2idx)
    print("类别数：", len(dataset.label2idx))

if __name__ == "__main__":
    test()
