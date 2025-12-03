import numpy as np
import open3d as o3d

def create_point_cloud(
    color_img: np.ndarray,
    depth_map: np.ndarray,
    normal_map: np.ndarray,
    mask: np.ndarray,
    mode: str = "linear",
    depth_scale: float = 0.5,
    fov_deg: float = 60.0  # <--- Added this argument back
) -> o3d.geometry.PointCloud:
    """
    Args:
        depth_map: Normalized depth/disparity (0..1)
        mode: 'linear' (Bas-Relief) or 'inverse' (Perspective)
        depth_scale: How 'thick' the object should be in meters
        fov_deg: Camera Field of View (lower = flatter telephoto look)
    """
    height, width = depth_map.shape

    # 1. Camera Intrinsics
    # Calculate focal length based on the provided FOV
    fov_rad = np.deg2rad(fov_deg)
    focal_length = (width / 2) / np.tan(fov_rad / 2)
    cx, cy = width / 2, height / 2

    # 2. Coordinate Grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # 3. Filter Mask
    valid = mask > 0.5
    z_raw = depth_map[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    # 4. Heuristic Z-Mapping
    if mode == "linear":
        # Bas-Relief
        z_metric = (1.0 - z_raw) * depth_scale + 1.0
    else:
        # Inverse (Perspective) - Best for human volume
        # We invert the disparity to get depth, then scale it
        # (z_raw + 0.1) prevents division by zero
        z_metric = 1.0 / (z_raw + 0.1) * depth_scale

    # 5. Back-Projection
    x_metric = (u_valid - cx) * z_metric / focal_length
    y_metric = (v_valid - cy) * z_metric / focal_length

    # Stack (Flip Y and Z for standard OpenGL/Blender coordinate system)
    points = np.stack((x_metric, -y_metric, -z_metric), axis=1)

    # 6. Colors & Normals
    colors = color_img[valid].astype(np.float64) / 255.0
    normals = normal_map[valid].astype(np.float64)

    # 7. Build Object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd