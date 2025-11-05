"""
Collision Detection Debug Script for SplatNav

This script adds comprehensive debugging to identify why paths go through obstacles.
Run this on your exp_room scene to diagnose the issue.
"""

import torch
import numpy as np
from pathlib import Path
from SFC.corridor_utils import SafeFlightCorridor
from splat.splat_utils import GSplatLoader
from splatplan.spline_utils import SplinePlanner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ CONFIGURATION (MODIFY FOR YOUR SCENE) ============
scene_name = 'exp_room'

# Your exp_room configuration
path_to_gsplat = "outputs/exp_room/exp_room/splatfacto/2025-10-22_174808/config.yml"

# Bounds (from your previous configuration)
lower = np.array([-2.30633705, -1.90576156, -2.72122359])
upper = np.array([1.18805204, 2.52680343, 0.15591765])

# Add margin
margin = 0.3
lower_bound = torch.tensor(lower + margin, device=device)
upper_bound = torch.tensor(upper - margin, device=device)

# Robot configuration - START WITH CONSERVATIVE VALUES
radius = 0.06  # 6cm robot radius
amax = 0.1
vmax = 0.1
resolution = 60  # Start with 60, can increase if needed

# Start/goal generation
xy_extent = min(upper[0] - lower[0], upper[1] - lower[1]) - 2 * margin
radius_config = 0.3 * xy_extent
radius_z = 0.05
mean_config = 0.5 * (lower + upper)
mean_config[2] = lower[2] + 0.75 * (upper[2] - lower[2])  # 75% height

print("=" * 80)
print(f"DEBUGGING COLLISION DETECTION FOR: {scene_name}")
print("=" * 80)

# ============ STAGE 0: LOAD GSPLAT ============
print("\n[STAGE 0] Loading Gaussian Splatting Model...")
print(f"  Path: {path_to_gsplat}")

gsplat = GSplatLoader(path_to_gsplat, device, filter_gaussians=True)

print(f"  ✓ Loaded {gsplat.means.shape[0]:,} Gaussians")
print(f"  ✓ Bounds: [{lower_bound.cpu().numpy()}] to [{upper_bound.cpu().numpy()}]")

# ============ STAGE 1: VOXEL GRID QUALITY CHECK ============
print("\n[STAGE 1] Voxel Grid Quality Check (MOST IMPORTANT!)")
print("-" * 80)

robot_config = {
    'radius': radius,
    'vmax': vmax,
    'amax': amax,
}

voxel_config = {
    'lower_bound': lower_bound,
    'upper_bound': upper_bound,
    'resolution': resolution,
}

spline_planner = SplinePlanner(spline_deg=6, device=device)
planner = SafeFlightCorridor(gsplat, robot_config, voxel_config, spline_planner, device, mode=1)

# Check voxel grid statistics
non_nav_count = planner.gsplat_voxel.non_navigable_grid.sum().item()
total_voxels = planner.gsplat_voxel.non_navigable_grid.numel()
occupancy = 100 * non_nav_count / total_voxels

print(f"  Grid resolution: {planner.gsplat_voxel.resolution}")
print(f"  Cell size: {planner.gsplat_voxel.cell_sizes.cpu().numpy()}")
print(f"  Robot radius: {radius:.4f} m")
print(f"  Robot radius in cells: {radius / planner.gsplat_voxel.cell_sizes[0].item():.2f}")
print(f"\n  Non-navigable voxels: {non_nav_count:,}")
print(f"  Total voxels: {total_voxels:,}")
print(f"  Occupancy: {occupancy:.1f}%")

# Diagnose
if occupancy < 5:
    print(f"\n  ❌ PROBLEM FOUND: Occupancy too low ({occupancy:.1f}%)")
    print(f"     → Voxel grid is too coarse to capture obstacles")
    print(f"     → SOLUTION: Increase resolution from {resolution} to {int(resolution * 1.5)}-{resolution * 2}")
elif occupancy > 95:
    print(f"\n  ❌ PROBLEM FOUND: Occupancy too high ({occupancy:.1f}%)")
    print(f"     → Robot radius too large or resolution too high")
    print(f"     → SOLUTION: Decrease radius to {radius * 0.7:.3f} or decrease resolution")
elif occupancy < 20:
    print(f"\n  ⚠️  WARNING: Occupancy low ({occupancy:.1f}%)")
    print(f"     → May miss some obstacles, consider increasing resolution")
elif occupancy > 60:
    print(f"\n  ⚠️  WARNING: Occupancy high ({occupancy:.1f}%)")
    print(f"     → May be too conservative, consider decreasing radius")
else:
    print(f"\n  ✓ Occupancy is good ({occupancy:.1f}%)")

# Save voxel mesh for visualization
print(f"\n  Saving voxel mesh to debug_voxel.obj...")
planner.gsplat_voxel.create_mesh('debug_voxel.obj')
print(f"  ✓ Saved! Open in MeshLab or Blender to visualize")

# ============ STAGE 2: TEST PATH GENERATION ============
print("\n[STAGE 2] Testing Path Generation")
print("-" * 80)

# Generate test start/goal
t_test = np.pi / 4  # 45 degrees
start = np.array([
    radius_config * np.cos(t_test),
    radius_config * np.sin(t_test),
    radius_z * np.sin(t_test)
]) + mean_config

goal = np.array([
    radius_config * np.cos(t_test + np.pi),
    radius_config * np.sin(t_test + np.pi),
    radius_z * np.sin(t_test + np.pi)
]) + mean_config

x = torch.tensor(start, device=device, dtype=torch.float32)
goal_t = torch.tensor(goal, device=device, dtype=torch.float32)

print(f"  Start: {start}")
print(f"  Goal:  {goal}")

# Generate path
output = planner.generate_path(x, goal_t)

if not output.get('feasible', False):
    print(f"\n  ❌ PROBLEM: No feasible path found!")
    print(f"     → Check if start/goal are in free space")
    print(f"     → Try different start/goal positions")
    exit(1)

print(f"  ✓ Path generated successfully")
print(f"  Time breakdown:")
print(f"    - A* initialization: {output.get('time_astar', 0):.4f}s")
print(f"    - Collision set: {output.get('time_collision_set', 0):.4f}s")
print(f"    - Polytope: {output.get('time_polytope', 0):.4f}s")
print(f"    - QP solver: {output.get('time_qp', 0):.4f}s")

# ============ STAGE 3: A* PATH VALIDATION ============
print("\n[STAGE 3] Validating A* Initial Path")
print("-" * 80)

# Re-run A* to get the initial path
astar_path = planner.generate_initialization(x, goal_t)

if astar_path is None:
    print(f"  ❌ A* returned None - no feasible path in voxel grid")
    exit(1)

print(f"  A* path length: {len(astar_path)} waypoints")

# Check if any A* waypoint is in occupied voxel
collision_count = 0
for i, point in enumerate(astar_path):
    indices = planner.voxel_grid.get_indices(torch.tensor(point, device=device))
    is_occupied = planner.voxel_grid.non_navigable_grid[indices[0], indices[1], indices[2]]

    if is_occupied:
        print(f"  ❌ A* waypoint {i}/{len(astar_path)} is in OCCUPIED voxel!")
        print(f"     Position: {point}")
        collision_count += 1

if collision_count > 0:
    print(f"\n  ❌ PROBLEM FOUND: {collision_count} A* waypoints in occupied voxels")
    print(f"     → Voxel grid has errors (go back to Stage 1)")
else:
    print(f"  ✓ All A* waypoints are in free voxels")

# ============ STAGE 4: TRAJECTORY COLLISION CHECK ============
print("\n[STAGE 4] Checking Final Trajectory for Collisions")
print("-" * 80)

trajectory = output['trajectory']
print(f"  Trajectory points: {len(trajectory)}")

# Sample 20 points along trajectory
sample_indices = np.linspace(0, len(trajectory) - 1, 20, dtype=int)
collision_points = []

for idx in sample_indices:
    pos = trajectory[idx, :3]

    # Find minimum distance to any Gaussian
    distances = torch.norm(planner.gsplat.means - torch.tensor(pos, device=device), dim=-1)
    min_dist = distances.min().item()
    min_idx = distances.argmin().item()

    # Check collision (using robot radius)
    if min_dist < radius:
        collision_points.append({
            'traj_idx': idx,
            'position': pos.cpu().numpy() if torch.is_tensor(pos) else pos,
            'min_distance': min_dist,
            'gaussian_idx': min_idx,
            'clearance_deficit': radius - min_dist
        })
        print(f"  ❌ Point {idx}: COLLISION at {pos}")
        print(f"     Min distance: {min_dist:.4f} m < radius {radius:.4f} m")
        print(f"     Deficit: {radius - min_dist:.4f} m")
    else:
        clearance = min_dist - radius
        if clearance < radius * 0.5:  # Less than 50% safety margin
            print(f"  ⚠️  Point {idx}: Low clearance {clearance:.4f} m")

if len(collision_points) > 0:
    print(f"\n  ❌ PROBLEM FOUND: {len(collision_points)}/{len(sample_indices)} sample points collide!")
    print(f"\n  Collision details:")
    for i, cp in enumerate(collision_points[:5]):  # Show first 5
        print(f"    {i+1}. Position {cp['position']}")
        print(f"       Distance: {cp['min_distance']:.4f} m (need {radius:.4f} m)")
        print(f"       Deficit: {cp['clearance_deficit']:.4f} m")
else:
    print(f"\n  ✓ No collisions detected in sampled trajectory points")

# ============ STAGE 5: GAUSSIAN FILTERING CHECK ============
print("\n[STAGE 5] Checking Gaussian Filtering Effects")
print("-" * 80)

# Check how many Gaussians are near the path
path_gaussians = 0
for idx in sample_indices:
    pos = trajectory[idx, :3]
    distances = torch.norm(planner.gsplat.means - torch.tensor(pos, device=device), dim=-1)
    nearby = (distances < radius * 3).sum().item()  # Within 3x robot radius
    path_gaussians += nearby

avg_nearby = path_gaussians / len(sample_indices)
print(f"  Average Gaussians within 3x radius of path: {avg_nearby:.1f}")

if avg_nearby < 5:
    print(f"  ⚠️  WARNING: Very few Gaussians near path")
    print(f"     → Filtering may be too aggressive")
    print(f"     → Try: GSplatLoader(path, device, filter_gaussians=False)")

# ============ SUMMARY ============
print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)

issues_found = []

if occupancy < 5:
    issues_found.append(f"Voxel occupancy too low ({occupancy:.1f}%) - increase resolution")
elif occupancy > 95:
    issues_found.append(f"Voxel occupancy too high ({occupancy:.1f}%) - decrease radius")

if collision_count > 0:
    issues_found.append(f"A* path has {collision_count} waypoints in occupied voxels")

if len(collision_points) > 0:
    issues_found.append(f"Final trajectory has {len(collision_points)} collision points")

if avg_nearby < 5:
    issues_found.append(f"Too few Gaussians near path ({avg_nearby:.1f}) - filtering too aggressive")

if len(issues_found) == 0:
    print("✓ No major issues detected!")
    print("  If you still see collisions visually, try:")
    print("  1. Increase robot radius slightly")
    print("  2. Increase voxel resolution")
    print("  3. Check visualization is using same robot radius")
else:
    print(f"Found {len(issues_found)} issue(s):")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")

print("\n" + "=" * 80)
print("RECOMMENDED ACTIONS:")
print("=" * 80)

if occupancy < 20:
    print(f"1. Increase resolution to {int(resolution * 1.5)}-{resolution * 2}")
if len(collision_points) > 0:
    print(f"2. Increase robot radius to {radius * 1.2:.3f} m")
if avg_nearby < 10:
    print(f"3. Disable Gaussian filtering: GSplatLoader(..., filter_gaussians=False)")
if collision_count > 0:
    print(f"4. Voxel grid has fundamental issues - check Stage 1 settings")

print("\nVisualization files created:")
print("  - debug_voxel.obj (voxel grid)")
print("  Open in MeshLab/Blender to inspect voxel quality")
