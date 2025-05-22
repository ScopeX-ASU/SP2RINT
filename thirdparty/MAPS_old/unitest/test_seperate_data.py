'''
read the forward field and adjoint field from the data
compare the calculated gradient and the gt gradient
'''
import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../MAPS")
)
sys.path.insert(0, project_root)
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from core.utils import print_stat

def test_gradient(filepath1, filepath2):
    # open the file
    with h5py.File(filepath1, "r") as f:
        keys = list(f.keys())
        # print(keys)
        # quit()
        # fwd_field = f["field_solutions-wl-1.55-port-in_port_1-mode-1-temp-300"][:]
        # adj_field = f["fields_adj-wl-1.55-port-in_port_1-mode-1"][:]
        # gradient = f["gradient"][:]
        source = f["adj_src"][:]
        max_val = np.max(source.real)
        eps_map = np.rot90(f["eps_map"][:])
        # plt.figure()
        # plt.imshow(np.rot90(fwd_field[-1, ...].real), cmap="RdBu")
        # plt.colorbar()
        # plt.title("Forward Field")
        # plt.savefig("./figs/forward_field.png")
        # plt.close()

        # plt.figure()
        # plt.imshow(np.rot90(adj_field[-1, ...].real), cmap="RdBu")
        # plt.colorbar()
        # plt.title("Adjoint Field")
        # plt.savefig("./figs/adjoint_field.png")
        # plt.close()
        
        # plt.figure()
        # plt.imshow(np.rot90(gradient.real), cmap="RdBu")
        # plt.colorbar()
        # plt.title("Gradient")
        # plt.savefig("./figs/gradient.png")
        # plt.close()

        # plt.figure()
        # plt.imshow(np.rot90(source.real), cmap="RdBu", vmax=max_val, vmin=-max_val)
        # plt.colorbar()
        # plt.title("Source")
        # plt.savefig("./figs/source.png")
        # plt.close()


        # plt.figure()

        # Plot eps_map in grayscale
        # plt.imshow(1 - eps_map, cmap="gray", interpolation="none", extent=[0, eps_map.shape[1], 0, eps_map.shape[0]])

        # Overlay source with transparency
        # plt.imshow(np.rot90(source.real), cmap="RdBu", alpha=0.6, vmin=-max_val, vmax=max_val) #, extent=[0, eps_map.shape[1], 0, eps_map.shape[0]])

        # plt.colorbar(label="Source Intensity")
        # plt.title("Overlaid Source and Eps Map")
        # plt.savefig("./figs/source_eps_overlay.png")
        # plt.close()

        # read the forward field and adjoint field
        forward_field = torch.from_numpy(f["field_solutions"][:]).to("cuda")
        adjoint_field = torch.from_numpy(f["fields_adj"][:]).to("cuda")
        # read the gradient
        gradient_gt_1 = torch.from_numpy(f["gradient"][:]).to("cuda")

        forward_field_1 = forward_field[-1, ...].contiguous()
        adjoint_field_1 = adjoint_field[-1, ...].contiguous()

        # calculate the gradient
        calculated_gradient_1 = (adjoint_field_1 * forward_field_1).real
        # print_stat(calculated_gradient)
        # print_stat(gradient)
        # quit()
        
        plt.figure()
        plt.imshow(calculated_gradient_1.cpu().detach().numpy(), cmap="RdBu")
        plt.colorbar()
        plt.title("Calculated Gradient")
        plt.savefig("./figs/grad_1.png")
        plt.close()

        plt.figure()
        plt.imshow(gradient_gt_1.cpu().detach().numpy(), cmap="RdBu")
        plt.colorbar()
        plt.title("GT Gradient")
        plt.savefig("./figs/grad_gt_1.png")
        plt.close()

    with h5py.File(filepath2, "r") as f:
        keys = list(f.keys())
        # read the forward field and adjoint field
        forward_field = torch.from_numpy(f["field_solutions"][:]).to("cuda")
        adjoint_field = torch.from_numpy(f["fields_adj"][:]).to("cuda")
        # read the gradient
        gradient_gt_2 = torch.from_numpy(f["gradient"][:]).to("cuda")
        total_gradient = torch.from_numpy(f["total_gradient"][:]).to("cuda")

        forward_field_2 = forward_field[-1, ...].contiguous()
        adjoint_field_2 = adjoint_field[-1, ...].contiguous()

        # calculate the gradient
        calculated_gradient_2 = (adjoint_field_2 * forward_field_2).real
        # print_stat(calculated_gradient)
        # print_stat(gradient)
        # quit()
        
        plt.figure()
        plt.imshow(calculated_gradient_2.cpu().detach().numpy(), cmap="RdBu")
        plt.colorbar()
        plt.title("Calculated Gradient")
        plt.savefig("./figs/grad_2.png")
        plt.close()

        plt.figure()
        plt.imshow(gradient_gt_2.cpu().detach().numpy(), cmap="RdBu")
        plt.colorbar()
        plt.title("GT Gradient")
        plt.savefig("./figs/grad_gt_2.png")
        plt.close()

        plt.figure()
        plt.imshow((gradient_gt_1 + gradient_gt_2).cpu().detach().numpy(), cmap="RdBu")
        plt.colorbar()
        plt.title("Calculated Gradient")
        plt.savefig("./figs/grad_sum.png")
        plt.close()

        plt.figure()
        plt.imshow(total_gradient.cpu().detach().numpy(), cmap="RdBu")
        plt.colorbar()
        plt.title("Calculated Gradient")
        plt.savefig("./figs/grad_total.png")
        plt.close()

if __name__ == "__main__":
    filepath1 = "/home/pingchua/projects/MAPS/data/fdfd/mdm/raw_test_hz_branch/mdm_id-0_opt_step_0-in_slice_1-1.55-Ez1-300.h5"
    filepath2 = "/home/pingchua/projects/MAPS/data/fdfd/mdm/raw_test_hz_branch/mdm_id-0_opt_step_0-in_slice_1-1.55-Ez2-300.h5"
    test_gradient(filepath1, filepath2)