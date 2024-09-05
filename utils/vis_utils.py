import os
import glob
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from utils.sh_utils import eval_sh
from einops import rearrange

def gridify():
    out_folder = "grids_objaverse"
    os.makedirs(out_folder, exist_ok=True)

    folder_paths = glob.glob("/scratch/shared/beegfs/stan/scaling_splatter_image/objaverse/*")
    folder_paths_test = sorted([fpath for fpath in folder_paths if "gt" not in fpath], key=lambda x: int(os.path.basename(x).split("_")[0]))

    # Initialize variables for grid dimensions
    num_examples_row = 6
    rows = num_examples_row
    num_per_ex = 2
    cols = num_examples_row * num_per_ex  # 6 * 2
    im_res = 128

    for im_idx in range(100):
        print(f"Doing frame {im_idx}")
        grid = np.zeros((rows*im_res, cols*im_res, 3), dtype=np.uint8)

        for f_idx, folder_path_test in enumerate(folder_paths_test[:num_examples_row*num_examples_row]):
            row_idx = f_idx // num_examples_row
            col_idx = f_idx % num_examples_row
            im_path = os.path.join(folder_path_test, "{:05d}.png".format(im_idx))
            im_path_gt = os.path.join(folder_path_test + "_gt", "{:05d}.png".format(im_idx))

            try:
                im = np.array(Image.open(im_path))
                im_gt = np.array(Image.open(im_path_gt))
                grid[row_idx * im_res: (row_idx+1) * im_res,
                     col_idx * num_per_ex * im_res: (col_idx * num_per_ex+1) * im_res, :] = im[:, :, :3]
                grid[row_idx * im_res: (row_idx+1) * im_res,
                     (col_idx * num_per_ex + 1) * im_res: (col_idx * num_per_ex + 2) * im_res, :] = im_gt[:, :, :3]
            except FileNotFoundError:
                pass
        im_out = Image.fromarray(grid)
        im_out.save(os.path.join(out_folder, "{:05d}.png".format(im_idx)))

def vis_image_preds(image_preds: dict, folder_out: str):
    """
    Visualises network's image predictions.
    Args:
        image_preds: a dictionary of xyz, opacity, scaling, rotation, features_dc and features_rest
    """
    image_preds_reshaped = {}
    ray_dirs = (image_preds["xyz"].detach().cpu() / torch.norm(image_preds["xyz"].detach().cpu(), dim=-1, keepdim=True)).reshape(128, 128, 3)

    for k, v in image_preds.items():
        image_preds_reshaped[k] = v
        if k == "xyz":
            # Normalize the xyz values between 0 and 1
            image_preds_reshaped[k] = (image_preds_reshaped[k] - torch.min(image_preds_reshaped[k])) / (
                torch.max(image_preds_reshaped[k]) - torch.min(image_preds_reshaped[k])
            )
        if k == "scaling":
            # Normalize the scaling values between 0 and 1
            image_preds_reshaped["scaling"] = (image_preds_reshaped["scaling"] - torch.min(image_preds_reshaped["scaling"])) / (
                torch.max(image_preds_reshaped["scaling"]) - torch.min(image_preds_reshaped["scaling"])
            )
        if k != "features_rest":
            image_preds_reshaped[k] = image_preds_reshaped[k].reshape(128, 128, -1).detach().cpu()
        else:
            image_preds_reshaped[k] = image_preds_reshaped[k].reshape(128, 128, 3, 3).detach().cpu().permute(0, 1, 3, 2)
        if k == "opacity":
            image_preds_reshaped[k] = image_preds_reshaped[k].expand(128, 128, 3)

    colours = torch.cat([image_preds_reshaped["features_dc"].unsqueeze(-1), image_preds_reshaped["features_rest"]], dim=-1)
    colours = eval_sh(1, colours, ray_dirs)

    # Ensure that colours are in the range [0, 1]
    colours = torch.clamp(colours, 0.0, 1.0)

    # Save the colours image
    plt.imsave(os.path.join(folder_out, "colours.png"), colours.numpy())

    # Save the opacity image
    plt.imsave(os.path.join(folder_out, "opacity.png"), image_preds_reshaped["opacity"].numpy())

    # Normalize and save the xyz image
    plt.imsave(os.path.join(folder_out, "xyz.png"), 
               (image_preds_reshaped["xyz"] * image_preds_reshaped["opacity"] + 1 - image_preds_reshaped["opacity"]).numpy())

    # Normalize and save the scaling image
    plt.imsave(os.path.join(folder_out, "scaling.png"), 
               (image_preds_reshaped["scaling"] * image_preds_reshaped["opacity"] + 1 - image_preds_reshaped["opacity"]).numpy())


if __name__ == "__main__":
    gridify()
