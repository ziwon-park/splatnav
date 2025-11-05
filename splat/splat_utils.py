import json
import torch
import numpy as np
from pathlib import Path
import open3d as o3d
import time

from ellipsoids.mesh_utils import create_gs_mesh
from ellipsoids.covariance_utils import quaternion_to_rotation_matrix
from ellipsoids.covariance_utils import compute_cov
from ns_utils.nerfstudio_utils import GaussianSplat, SH2RGB

class GSplatLoader():
    def __init__(self, gsplat_location, device, filter_gaussians=True):
        self.device = device

        if isinstance(gsplat_location, str):
            # Check file extension
            if gsplat_location.endswith('.json'):
                self.load_gsplat_from_json(gsplat_location)
            elif gsplat_location.endswith('.splat'):
                self.load_gsplat_from_splat(gsplat_location)
            elif gsplat_location.endswith('.ply'):
                self.load_gsplat_from_ply(gsplat_location)
            else:
                raise ValueError('Unknown file format. Supported: .json, .splat, .ply')
        elif isinstance(gsplat_location, Path):
            self.load_gsplat_from_nerfstudio(gsplat_location)
        else:
            raise ValueError('GSplat file must be a Path (.yml) or string (.json/.splat/.ply).')

        # Apply filtering after loading
        if filter_gaussians:
            self.filter_low_quality_gaussians()
        
    def load_gsplat_from_nerfstudio(self, gsplat_location):

        self.splat = GaussianSplat(gsplat_location,
                    test_mode= "inference",
                    dataset_mode = 'train',
                    device = self.device)

        self.means = self.splat.pipeline.model.means.detach().clone()
        self.rots = self.splat.pipeline.model.quats.detach().clone()
        self.scales = self.splat.pipeline.model.scales.detach().clone()
        self.scales = torch.exp(self.scales)

        self.covs_inv = compute_cov(self.rots, 1 / self.scales)
        self.covs = compute_cov(self.rots, self.scales)

        self.colors = SH2RGB(self.splat.pipeline.model.features_dc.detach().clone())

        self.opacities = torch.sigmoid(self.splat.pipeline.model.opacities.detach().clone())

        print(f'There are {self.means.shape[0]} Gaussians in the GSplat model')

        return

    def load_gsplat_from_json(self, gsplat_location):

        with open(gsplat_location, 'r') as f:
            data = json.load(f)
        
        keys = ['means', 'rotations', 'colors', 'opacities', 'scalings']
        tensors = {}

        # Measure time for loading tensors
        start_time = time.time()
        for key in keys:
            tensors[key] = torch.tensor(data[key]).to(dtype=torch.float32, device=self.device)
        print(f"Loading tensors took {time.time() - start_time:.4f} seconds")
        
        # Measure time for setting attributes
        start_time = time.time()
        self.means = tensors['means']
        self.rots = tensors['rotations']
        self.colors = tensors['colors']
        self.opacities = tensors['opacities']
        self.scales = tensors['scalings']

        print(f"Setting attributes took {time.time() - start_time:.4f} seconds")

        # Print tensor sizes
        print(f"Opacities tensor size: {self.opacities.size()}")
        print(f"Scales tensor size: {self.scales.size()}")

        # Measure time for normalization
        self.opacities = torch.sigmoid(self.opacities)
        self.scales = torch.exp(self.scales)

        # Measure time for computing Sigma inverse
        self.covs_inv = compute_cov(self.rots, 1. / self.scales)
        self.covs = compute_cov(self.rots, self.scales)

        return

    def load_gsplat_from_splat(self, gsplat_location):
        """
        Load from .splat binary format (Antimatter15/splat format).
        Each Gaussian: 14 floats (56 bytes)
        - position (3), scale (3), color (4 RGBA), rotation (4 quaternion)
        """
        import struct

        with open(gsplat_location, 'rb') as f:
            data = f.read()

        # Each Gaussian is 14 floats (4 bytes each) = 56 bytes
        num_gaussians = len(data) // (14 * 4)

        means = []
        scales = []
        colors = []
        opacities = []
        rots = []

        for i in range(num_gaussians):
            offset = i * 14 * 4
            gaussian = struct.unpack('14f', data[offset:offset + 56])

            means.append(gaussian[0:3])      # xyz position
            scales.append(gaussian[3:6])     # scale (log space)
            colors.append(gaussian[6:9])     # rgb
            opacities.append(gaussian[9])    # alpha
            rots.append(gaussian[10:14])     # quaternion wxyz

        # Convert to tensors
        self.means = torch.tensor(means, dtype=torch.float32, device=self.device)
        self.scales = torch.tensor(scales, dtype=torch.float32, device=self.device)
        self.colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
        self.opacities = torch.tensor(opacities, dtype=torch.float32, device=self.device).unsqueeze(-1)
        self.rots = torch.tensor(rots, dtype=torch.float32, device=self.device)

        # Apply transformations
        self.opacities = torch.sigmoid(self.opacities)
        self.scales = torch.exp(self.scales)

        # Compute covariances
        self.covs_inv = compute_cov(self.rots, 1. / self.scales)
        self.covs = compute_cov(self.rots, self.scales)

        print(f'[.splat] Loaded {num_gaussians:,} Gaussians from {gsplat_location}')

        return

    def load_gsplat_from_ply(self, gsplat_location):
        """
        Load from .ply format (standard Gaussian Splatting PLY with extended attributes).
        Uses plyfile library to parse.
        """
        try:
            from plyfile import PlyData
        except ImportError:
            raise ImportError('plyfile not installed. Run: pip install plyfile')

        plydata = PlyData.read(gsplat_location)
        vertex = plydata['vertex']

        # Extract position
        means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)

        # Extract scale (f_dc_0, f_dc_1, f_dc_2 or scale_0, scale_1, scale_2)
        try:
            scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
        except:
            # Try alternative naming
            scales = np.stack([vertex['scale_x'], vertex['scale_y'], vertex['scale_z']], axis=1)

        # Extract colors (f_dc_0, f_dc_1, f_dc_2 or r, g, b)
        try:
            colors = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)
        except:
            colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1) / 255.0

        # Extract opacity
        try:
            opacities = vertex['opacity']
        except:
            opacities = vertex['alpha']

        # Extract rotation (quaternion)
        try:
            rots = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
        except:
            rots = np.stack([vertex['qw'], vertex['qx'], vertex['qy'], vertex['qz']], axis=1)

        # Convert to tensors
        self.means = torch.tensor(means, dtype=torch.float32, device=self.device)
        self.scales = torch.tensor(scales, dtype=torch.float32, device=self.device)
        self.colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
        self.opacities = torch.tensor(opacities, dtype=torch.float32, device=self.device).unsqueeze(-1)
        self.rots = torch.tensor(rots, dtype=torch.float32, device=self.device)

        # Apply transformations (if needed)
        self.opacities = torch.sigmoid(self.opacities)
        self.scales = torch.exp(self.scales)

        # Compute covariances
        self.covs_inv = compute_cov(self.rots, 1. / self.scales)
        self.covs = compute_cov(self.rots, self.scales)

        print(f'[.ply] Loaded {len(means):,} Gaussians from {gsplat_location}')

        return

    def filter_low_quality_gaussians(self, opacity_threshold=0.1, scale_percentile=95, distance_percentile=99):
        """
        Filter out low-quality Gaussians to reduce memory usage and improve quality.

        Args:
            opacity_threshold: Remove Gaussians with opacity below this value (default: 0.1)
            scale_percentile: Remove Gaussians with max scale above this percentile (default: 95)
            distance_percentile: Remove Gaussians beyond this distance percentile from center (default: 99)
        """
        num_original = self.means.shape[0]

        # 1. Opacity filter (most effective)
        opacity_mask = self.opacities.squeeze() >= opacity_threshold

        # 2. Scale filter (remove outliers)
        max_scales = torch.max(self.scales, dim=-1)[0]  # Max scale per Gaussian
        scale_threshold = torch.quantile(max_scales, scale_percentile / 100.0)
        scale_mask = max_scales <= scale_threshold

        # 3. Distance filter (remove distant Gaussians)
        center = torch.median(self.means, dim=0)[0]  # Scene center
        distances = torch.norm(self.means - center, dim=-1)
        distance_threshold = torch.quantile(distances, distance_percentile / 100.0)
        distance_mask = distances <= distance_threshold

        # Combine all filters
        combined_mask = opacity_mask & scale_mask & distance_mask

        # Apply filter
        self.means = self.means[combined_mask]
        self.rots = self.rots[combined_mask]
        self.scales = self.scales[combined_mask]
        self.colors = self.colors[combined_mask]
        self.opacities = self.opacities[combined_mask]
        self.covs = self.covs[combined_mask]
        self.covs_inv = self.covs_inv[combined_mask]

        num_filtered = self.means.shape[0]
        reduction_pct = 100 * (1 - num_filtered / num_original)

        print(f'[Filter] Reduced Gaussians: {num_original:,} → {num_filtered:,} ({reduction_pct:.1f}% removed)')
        print(f'  - Opacity < {opacity_threshold}: {(~opacity_mask).sum().item():,}')
        print(f'  - Scale > {scale_percentile}%ile: {(~scale_mask).sum().item():,}')
        print(f'  - Distance > {distance_percentile}%ile: {(~distance_mask).sum().item():,}')

        return combined_mask

    def save_mesh(self, filepath, bounds=None, res=4):
        if bounds is not None:
            mask = torch.all((self.means - bounds[:, 0] >= 0) & (bounds[:, 1] - self.means >= 0), dim=-1)
            means = self.means[mask]
            rots = self.rots[mask]
            scales = self.scales[mask]
            colors = self.colors[mask]
        else:
            means = self.means
            rots = self.rots
            scales = self.scales
            colors = self.colors

        scene = create_gs_mesh(means.cpu().numpy(), quaternion_to_rotation_matrix(rots).cpu().numpy(), scales.cpu().numpy(), colors.cpu().numpy(), res=res, transform=None, scale=None)
        success = o3d.io.write_triangle_mesh(filepath, scene, print_progress=True)

        return success


# Loader for GSplat means 
class PointCloudLoader(GSplatLoader):
    def __init__(self, device):
        self.device = device

    def initialize_attributes(self, means):
        self.means = means.to(self.device)
        return
    
# The purpose of this loader is to run toy examples and for figures.
class DummyGSplatLoader(GSplatLoader):
    def __init__(self, device):
        self.device = device

    def initialize_attributes(self, means, rots, scales, colors=None):
        self.means = means.to(self.device)
        self.rots = rots.to(self.device)
        self.scales = scales.to(self.device)

        self.cov_inv = compute_cov(self.rots, 1 / self.scales)
        self.covs = compute_cov(self.rots, self.scales)

        if colors is not None:
            self.colors = colors.to(self.device)
        else:
            self.colors = 0.5*torch.ones(means.shape[0], 3).to(self.device)

        return