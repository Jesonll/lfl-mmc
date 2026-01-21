import os
import numpy as np
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from .lower_layer import LowerLayerSolver

class GCodeExporter:
    @staticmethod
    def export(filepath, all_paths, stay_points, allocations, config):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        v_mark = config['machine']['galvo_mark_speed'] * 60
        v_jump = config['machine']['galvo_jump_speed'] * 60
        plat_speed = config['machine']['platform_speed'] * 60
        
        # 1. Identify Used Stations
        used_indices = np.unique(allocations)
        if len(used_indices) == 0:
            print("No paths assigned to any station.")
            return

        # 2. Sort Stations by TSP (Nearest Neighbor from 0,0)
        active_sps = stay_points[used_indices]
        pts_with_home = np.vstack(([0,0], active_sps))
        dists = squareform(pdist(pts_with_home))
        
        visited = [False] * len(pts_with_home)
        curr = 0
        visited[0] = True
        station_order = []
        
        for _ in range(len(active_sps)):
            dists[curr][visited] = np.inf # Mask visited
            next_node = np.argmin(dists[curr])
            
            # Map back to original indices
            # next_node is index in pts_with_home (1-based for stations)
            real_idx = used_indices[next_node - 1]
            station_order.append(real_idx)
            
            visited[next_node] = True
            curr = next_node
            
        print(f"[Export] Generating G-Code for {len(station_order)} stations...")

        with open(filepath, 'w') as f:
            f.write(f"; Dual-Mode Laser Optimization\n; Date: {timestamp}\n")
            f.write("G21 ; Units MM (assuming input scaled to M)\nG90\n\n")

            for step_num, sp_idx in enumerate(station_order):
                sp = stay_points[sp_idx]
                cluster_indices = np.where(allocations == sp_idx)[0]
                cluster_paths = [all_paths[i] for i in cluster_indices]
                
                f.write(f"; =========================================\n")
                f.write(f"; STATION {sp_idx} (Sequence #{step_num+1})\n")
                f.write(f"; Location: X{sp[0]:.4f} Y{sp[1]:.4f}\n")
                f.write(f"; Vectors: {len(cluster_paths)}\n")
                f.write(f"; =========================================\n")
                
                # Platform Move
                f.write(f"G0 X{sp[0]:.4f} Y{sp[1]:.4f} F{plat_speed:.1f}\n")
                f.write("M0 ; Settle Platform\n")
                
                # Solve Galvo Path
                solver = LowerLayerSolver(cluster_paths, sp, config)
                _, sequence = solver.solve()
                
                current_g = np.array([0., 0.])
                
                for p_idx, rev in sequence:
                    path = cluster_paths[p_idx]
                    local_pts = path.points - sp
                    
                    start = local_pts[-1] if rev else local_pts[0]
                    end = local_pts[0] if rev else local_pts[-1]
                    
                    # Jump
                    f.write(f"G0 X{start[0]:.4f} Y{start[1]:.4f} F{v_jump:.1f} ; Jump\n")
                    
                    # Cut
                    f.write("M3\n")
                    draw_pts = local_pts[::-1] if rev else local_pts
                    for pt in draw_pts:
                        f.write(f"G1 X{pt[0]:.4f} Y{pt[1]:.4f} F{v_mark:.1f}\n")
                    f.write("M5\n")
                    
                    current_g = end
                f.write("\n")

            f.write("M2 ; End\n")
        print(f"[Export] Saved to {filepath}")