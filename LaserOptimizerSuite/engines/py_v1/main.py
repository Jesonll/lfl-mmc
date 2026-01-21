import argparse
import json
import os
import sys
import numpy as np
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.svg_loader import SVGLoader
from core.upper_layer import UpperLayerOptimizer
from core.visualization import plot_results
from core.exporter import GCodeExporter

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def normalize_paths(paths, target_width, target_height):
    """
    Scales and shifts the pattern to fit within target_width x target_height.
    Preserves Aspect Ratio.
    Moves bottom-left to (0,0).
    """
    if not paths: return paths
    
    print(f"[Process] Normalizing pattern to fit {target_width}m x {target_height}m...")
    
    # 1. Get current bounds
    all_pts = np.vstack([p.points for p in paths])
    min_xy = np.min(all_pts, axis=0)
    max_xy = np.max(all_pts, axis=0)
    
    curr_w = max_xy[0] - min_xy[0]
    curr_h = max_xy[1] - min_xy[1]
    
    # Avoid division by zero
    if curr_w < 1e-9: curr_w = 1.0
    if curr_h < 1e-9: curr_h = 1.0
    
    # 2. Calculate Scale Factor (Maintain Aspect Ratio)
    scale_x = target_width / curr_w
    scale_y = target_height / curr_h
    scale = min(scale_x, scale_y)
    
    print(f"          Original Size: {curr_w:.2f} x {curr_h:.2f}")
    print(f"          Scale Factor: {scale:.4f}")
    
    # 3. Apply Scale and Shift
    for p in paths:
        # Shift to 0,0 then Scale
        p.points = (p.points - min_xy) * scale
        # Recalculate length
        p.length = np.sum(np.sqrt(np.sum(np.diff(p.points, axis=0)**2, axis=1)))
        
    return paths

def tile_paths(paths, nx, ny, spacing_x=0.05, spacing_y=0.05):
    """Duplicates the pattern into a grid"""
    tiled_paths = []
    pid_counter = 0
    
    # Calculate bounds of original
    all_pts = np.vstack([p.points for p in paths])
    min_xy = np.min(all_pts, axis=0)
    max_xy = np.max(all_pts, axis=0)
    w = max_xy[0] - min_xy[0]
    h = max_xy[1] - min_xy[1]
    
    step_x = w + spacing_x
    step_y = h + spacing_y
    
    print(f"[Process] Tiling input {nx}x{ny}...")
    
    for iy in range(ny):
        for ix in range(nx):
            offset = np.array([ix * step_x, iy * step_y])
            for p in paths:
                new_pts = p.points + offset
                new_p = copy.deepcopy(p)
                new_p.id = pid_counter
                new_p.points = new_pts
                tiled_paths.append(new_p)
                pid_counter += 1
                
    return tiled_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="Path to SVG")
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--output', type=str, default='output.gcode')
    parser.add_argument('--tile', type=int, default=1, help="Tile pattern NxN times")
    
    # NEW ARGUMENT: Target Size
    parser.add_argument('--target_size', type=float, nargs=2, metavar=('W', 'H'),
                        help="Normalize input to Width Height (in Meters). E.g. --target_size 0.15 0.15")
    
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("Config not found.")
        return
    config = load_config(args.config)

    # 1. Load (Raw Scale)
    # We set internal scale to 1.0 here because we will normalize explicitly later
    if args.input and os.path.exists(args.input):
        paths = SVGLoader.load_svg(args.input, scale=1.0, sampling_resolution=50.0)
    else:
        paths = SVGLoader.generate_mock_data()

    if not paths:
        print("Error: No paths loaded.")
        return

    # 2. Normalize (User Request)
    if args.target_size:
        target_w, target_h = args.target_size
        paths = normalize_paths(paths, target_w, target_h)
    else:
        # Fallback to config scaling if no target size provided
        cfg_scale = config['process']['svg_scale_factor']
        if abs(cfg_scale - 1.0) > 1e-6:
            print(f"[Process] Applying config scale: {cfg_scale}")
            for p in paths:
                p.points *= cfg_scale
                p.length *= cfg_scale

    # 3. Tile
    if args.tile > 1:
        paths = tile_paths(paths, args.tile, args.tile, spacing_x=0.02, spacing_y=0.02)

    # 4. Optimize
    optimizer = UpperLayerOptimizer(paths, config)
    best_stay_points, allocations, best_time, history = optimizer.optimize()

    print(f"\nOptimization Complete. Best Time: {best_time:.4f}s")
    
    GCodeExporter.export(args.output, paths, best_stay_points, allocations, config)
    plot_results(paths, best_stay_points, allocations, history, config)

if __name__ == "__main__":
    main()