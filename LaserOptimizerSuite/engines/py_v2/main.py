import argparse
import json
import os
import sys
import numpy as np
import copy
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.svg_loader import SVGLoader
from core.upper_layer import UpperLayerOptimizer
from core.lower_layer import LowerLayerSolver
from core.visualization import plot_results
from core.exporter import GCodeExporter

def load_config(path):
    with open(path, 'r') as f: return json.load(f)

def normalize_paths(paths, target_width, target_height):
    if not paths: return paths
    print(f"[Process] Normalizing pattern to fit {target_width}m x {target_height}m...")
    
    # 1. Flip Y (Machine Coords: Y Up)
    # The SVG input (screen coords) has Y Down.
    for p in paths:
        p.points[:, 1] = -p.points[:, 1]

    # 2. Shift to (0,0)
    all_pts = np.vstack([p.points for p in paths])
    min_xy = np.min(all_pts, axis=0)
    for p in paths:
        p.points -= min_xy # Moves lowest point to 0

    # 3. Scale
    all_pts = np.vstack([p.points for p in paths]) # Re-calc bounds after shift
    max_xy = np.max(all_pts, axis=0) # min is now 0,0
    curr_w = max_xy[0]
    curr_h = max_xy[1]
    
    if curr_w < 1e-9: curr_w = 1.0
    if curr_h < 1e-9: curr_h = 1.0
    
    scale = min(target_width / curr_w, target_height / curr_h)
    print(f"          Scale Factor: {scale:.4f}")
    
    for p in paths:
        p.points *= scale
        p.length = np.sum(np.sqrt(np.sum(np.diff(p.points, axis=0)**2, axis=1)))
        
    return paths

def tile_paths(paths, nx, ny, spacing_x=0.05, spacing_y=0.05):
    tiled_paths = []
    pid_counter = 0
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

def calculate_detailed_metrics(paths, stay_points, allocations, sequence, config):
    t_plat, t_jump, t_mark = 0.0, 0.0, 0.0
    v_plat = config['machine']['platform_speed']
    v_mark = config['machine']['galvo_mark_speed']
    v_jump = config['machine']['galvo_jump_speed']
    current_plat = np.array([0.0, 0.0]) 
    
    for sp_idx in sequence:
        sp_pos = stay_points[sp_idx]
        t_plat += np.linalg.norm(sp_pos - current_plat) / v_plat + 0.1 
        current_plat = sp_pos
        cluster_indices = np.where(allocations == sp_idx)[0]
        cluster_paths = [paths[i] for i in cluster_indices]
        solver = LowerLayerSolver(cluster_paths, sp_pos, config)
        _, micro_sequence = solver.solve()
        
        current_galvo = np.array([0.0, 0.0])
        for p_idx, reverse in micro_sequence:
            path = cluster_paths[p_idx]
            local_pts = path.points - sp_pos
            start = local_pts[-1] if reverse else local_pts[0]
            end = local_pts[0] if reverse else local_pts[-1]
            t_jump += np.linalg.norm(start - current_galvo) / v_jump
            t_mark += path.length / v_mark
            current_galvo = end
    return t_plat, t_jump, t_mark

def main():
    wall_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="Path to SVG")
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--output', type=str, default='output.gcode')
    parser.add_argument('--tile', type=int, default=1)
    parser.add_argument('--target_size', type=float, nargs=2)
    parser.add_argument('--segment_len', type=float, default=0.05)
    args = parser.parse_args()

    if not os.path.exists(args.config): return
    config = load_config(args.config)

    print("[1/5] Loading SVG...")
    if args.input and os.path.exists(args.input):
        paths = SVGLoader.load_svg(args.input, scale=1.0, segment_len_mm=args.segment_len)
    else: return

    if args.target_size:
        paths = normalize_paths(paths, args.target_size[0], args.target_size[1])
    else:
        cfg_scale = config['process']['svg_scale_factor']
        # Also ensure flip happens in non-normalize mode
        for p in paths: p.points[:, 1] = -p.points[:, 1]
        for p in paths: p.points *= cfg_scale

    if args.tile > 1:
        paths = tile_paths(paths, args.tile, args.tile, spacing_x=0.02, spacing_y=0.02)

    print(f"[3/5] Optimizing {len(paths)} vectors (PSO)...")
    opt_start = time.time()
    optimizer = UpperLayerOptimizer(paths, config)
    best_stay_points, allocations, sequence, _, _ = optimizer.optimize()
    print(f"      Optimization took {time.time() - opt_start:.2f}s")

    t_plat, t_jump, t_mark = calculate_detailed_metrics(paths, best_stay_points, allocations, sequence, config)
    total_physical_time = t_plat + t_jump + t_mark
    
    total_points = sum(len(p.points) for p in paths)
    total_length = sum(p.length for p in paths)
    avg_seg_len_mm = (total_length * 1000.0 / total_points) if total_points > 0 else 0.0

    metrics = {
        "engine": "py_v2",
        "time_compute_sec": time.time() - wall_start,
        "time_total_machine_sec": total_physical_time,
        "time_platform_sec": t_plat,
        "time_jump_sec": t_jump,
        "time_mark_sec": t_mark,
        "geo_resolution_mm": avg_seg_len_mm
    }
    
    metrics_path = args.output + ".json"
    with open(metrics_path, 'w') as f: json.dump(metrics, f, indent=4)

    GCodeExporter.export(args.output, paths, best_stay_points, allocations, config, sequence=sequence, total_time=total_physical_time)
    plot_results(paths, best_stay_points, allocations, [], config, sequence=sequence)

if __name__ == "__main__":
    main()