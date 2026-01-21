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
    if not paths: return paths
    print(f"[Process] Normalizing pattern to fit {target_width}m x {target_height}m...")
    all_pts = np.vstack([p.points for p in paths])
    min_xy = np.min(all_pts, axis=0)
    max_xy = np.max(all_pts, axis=0)
    curr_w = max_xy[0] - min_xy[0]
    curr_h = max_xy[1] - min_xy[1]
    if curr_w < 1e-9: curr_w = 1.0
    if curr_h < 1e-9: curr_h = 1.0
    scale = min(target_width / curr_w, target_height / curr_h)
    print(f"          Scale Factor: {scale:.4f}")
    for p in paths:
        p.points = (p.points - min_xy) * scale
        p.length = np.sum(np.sqrt(np.sum(np.diff(p.points, axis=0)**2, axis=1)))
    return paths

def tile_paths(paths, nx, ny, spacing_x=0.05, spacing_y=0.05):
    tiled_paths = []
    pid_counter = 0
    all_pts = np.vstack([p.points for p in paths])
    min_xy = np.min(all_pts, axis=0)
    max_xy = np.max(all_pts, axis=0)
    w, h = max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]
    step_x, step_y = w + spacing_x, h + spacing_y
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
    parser.add_argument('--tile', type=int, default=1)
    parser.add_argument('--target_size', type=float, nargs=2)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("Config not found.")
        return
    config = load_config(args.config)

    if args.input and os.path.exists(args.input):
        paths = SVGLoader.load_svg(args.input, scale=1.0, sampling_resolution=50.0)
    else:
        paths = SVGLoader.generate_mock_data()
    
    if not paths: return

    if args.target_size:
        paths = normalize_paths(paths, args.target_size[0], args.target_size[1])
    else:
        cfg_scale = config['process']['svg_scale_factor']
        if abs(cfg_scale - 1.0) > 1e-6:
            for p in paths: p.points *= cfg_scale

    if args.tile > 1:
        paths = tile_paths(paths, args.tile, args.tile, spacing_x=0.02, spacing_y=0.02)

    optimizer = UpperLayerOptimizer(paths, config)
    
    # --- CHANGED: Unpack 5 values (including sequence) ---
    best_stay_points, allocations, sequence, best_time, history = optimizer.optimize()

    print(f"\nOptimization Complete. Best Time: {best_time:.4f}s")
    
    # --- CHANGED: Pass sequence to Exporter and Visualizer ---
    GCodeExporter.export(args.output, paths, best_stay_points, allocations, config, sequence=sequence)
    plot_results(paths, best_stay_points, allocations, history, config, sequence=sequence)

if __name__ == "__main__":
    main()