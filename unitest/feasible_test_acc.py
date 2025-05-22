import re
import numpy as np
import csv

# Path to the .log file
log_file_path = "/home/pingchua/projects/MetaONN/log/fmnist/meta_cnn/Benchmark/run-46_meta_cnn_fmnist_lr-0.002_pd-2_enc-phase_lam- 0.850_dz- 4.000_ps- 0.300_c-metaline_training.log"

# Regular expression for "Test set" line
pattern = r"Test set: Average loss: ([\deE\.+-]+), Accuracy: (\d+)/(\d+) \(([\d\.]+)%\)"

# Lists to collect loss and accuracy
loss_list = []
accuracy_list = []

# Parse log
with open(log_file_path, 'r') as file:
    for line in file:
        if 'Test set: Average loss' in line:
            match = re.search(pattern, line)
            if match:
                loss = float(match.group(1))
                accuracy_percent = float(match.group(4))

                loss_list.append(loss)
                accuracy_list.append(accuracy_percent)

# print("this is the length loss_list:" , len(loss_list))
# print("this is the length accuracy_list:" , len(accuracy_list))
print("this is the loss_list:" , len(loss_list))
print("this is the accuracy_list:" , len(accuracy_list))
LR_loss_array = np.array(loss_list)[1::3]
HR_loss_array = np.array(loss_list)[2::3]
LR_accuracy_array = np.array(accuracy_list)[1::3]
HR_accuracy_array = np.array(accuracy_list)[2::3]

loss_degrade = (HR_loss_array - LR_loss_array).mean()
accuracy_degrade = (LR_accuracy_array - HR_accuracy_array).mean()

print(f'Mean CE Loss degrade: {loss_degrade:.4e}')
print(f'Mean Accuracy degrade: {accuracy_degrade:.2f}%')

with open("./unitest/HR_accuracy_array_metaline.csv", "w") as f:
    writer = csv.writer(f)
    for i in range(len(HR_accuracy_array)):
        writer.writerow([i + 1, HR_accuracy_array[i]])