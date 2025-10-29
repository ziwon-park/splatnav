import json
import torch
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
            self.load_gsplat_from_json(gsplat_location)
        elif isinstance(gsplat_location, Path):
            self.load_gsplat_from_nerfstudio(gsplat_location)
        else:
            raise ValueError('GSplat file must be either a .json or .yml file.')

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