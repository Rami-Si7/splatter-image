import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import focal2fov

def render_predicted(pc: dict, 
                     world_view_transform,
                     full_proj_transform,
                     camera_center,
                     bg_color: torch.Tensor, 
                     cfg, 
                     scaling_modifier=1.0, 
                     override_color=None,
                     focals_pixels=None):
    """
    Render the scene as specified by pc dictionary. 
    Returns both the rendered image and the depth map.
    """

    # Create zero tensor for 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc["xyz"], dtype=pc["xyz"].dtype, requires_grad=True, device=pc["xyz"].device)

    if focals_pixels is None:
        tanfovx = math.tan(cfg.data.fov * np.pi / 360)
        tanfovy = math.tan(cfg.data.fov * np.pi / 360)
    else:
        tanfovx = math.tan(0.5 * focal2fov(focals_pixels[0].item(), cfg.data.training_resolution))
        tanfovy = math.tan(0.5 * focal2fov(focals_pixels[1].item(), cfg.data.training_resolution))

    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(
        image_height=int(cfg.data.training_resolution),
        image_width=int(cfg.data.training_resolution),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=cfg.model.max_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc["xyz"]  # The 3D positions of the Gaussians
    means2D = screenspace_points
    opacity = pc["opacity"]

    # If precomputed 3D covariance is provided, use it.
    scales = pc["scaling"]
    rotations = pc["rotation"]

    # Handling SHs or Precomputed Colors
    shs = None
    colors_precomp = None
    if override_color is None:
        if "features_rest" in pc.keys():
            shs = torch.cat([pc["features_dc"], pc["features_rest"]], dim=1).contiguous()
        else:
            shs = pc["features_dc"]
    else:
        colors_precomp = override_color

    # Ensure either SHs or colors_precomp is provided
    if shs is None and colors_precomp is None:
        raise Exception('Please provide either SHs or precomputed colors!')

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,  # Pass SHs if available
        colors_precomp=colors_precomp,  # Or pass precomputed colors
        opacities=opacity,
        scales=scales,
        rotations=rotations
    )

    # Depth map generation
    # Extract the Z-coordinates (depth values)
    depth_values = means3D[:, 2]

    # Normalize the depth values
    depth_min = torch.min(depth_values)
    depth_max = torch.max(depth_values)
    normalized_depth = (depth_values - depth_min) / (depth_max - depth_min)

    # Reshape the depth values accordingly (depth should be a 2D map, not 3D)
    image_height, image_width = rendered_image.shape[-2], rendered_image.shape[-1]
    depth_map = normalized_depth.view(image_height, image_width)

    # Save and display depth map as an image
    # plt.imsave('/content/depth_map.png', depth_map.cpu().numpy(), cmap='gray')
    # display(Image.open('/content/depth_map.png'))

    return {
        "render": rendered_image,
        "depth_map": depth_map,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    }
