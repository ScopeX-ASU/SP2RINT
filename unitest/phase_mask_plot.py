import torch
import matplotlib.pyplot as plt
import numpy as np
import random

if __name__ == "__main__":
    # Create a 2D tensor with random values
    tm = torch.zeros(32, 32)
    for idx in range(4):
        for i in range(32):
            random_phaes = random.uniform(0, 2 * np.pi)
            tm[i, i] = random_phaes
        plt.figure()
        plt.imshow(tm.numpy())
        plt.savefig(f"./figs/phase_mask_{idx}.png")
        plt.close()