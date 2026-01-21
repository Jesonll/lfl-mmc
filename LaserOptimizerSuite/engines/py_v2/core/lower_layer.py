import numpy as np
from typing import List, Tuple
from .svg_loader import LaserPath

class LowerLayerSolver:
    def __init__(self, paths: List[LaserPath], origin: np.ndarray, config: dict):
        self.paths = paths
        self.origin = origin
        self.config = config

    def solve(self) -> Tuple[float, List[Tuple[int, bool]]]:
        if not self.paths:
            return 0.0, []

        v_mark = self.config['machine']['galvo_mark_speed']
        v_jump = self.config['machine']['galvo_jump_speed']

        # 1. Fixed Cost: Marking Time
        marking_dist = sum(p.length for p in self.paths)
        marking_time = marking_dist / v_mark

        # 2. Variable Cost: Jump Time (Greedy Nearest Neighbor with Direction)
        n = len(self.paths)
        unvisited = set(range(n))
        
        # Start at stay point center (0,0 relative)
        current_pos = np.array([0.0, 0.0]) 
        sequence = [] 
        total_jump_dist = 0.0

        while unvisited:
            best_idx = -1
            best_dist = float('inf')
            best_reverse = False

            for idx in unvisited:
                path = self.paths[idx]
                # Coords relative to stay point
                p_start = path.points[0] - self.origin
                p_end = path.points[-1] - self.origin

                # Cost to hit Start
                d_start = np.linalg.norm(p_start - current_pos)
                # Cost to hit End (and traverse reverse)
                d_end = np.linalg.norm(p_end - current_pos)

                if d_start < best_dist:
                    best_dist = d_start
                    best_idx = idx
                    best_reverse = False
                
                if d_end < best_dist:
                    best_dist = d_end
                    best_idx = idx
                    best_reverse = True

            unvisited.remove(best_idx)
            sequence.append((best_idx, best_reverse))
            total_jump_dist += best_dist

            # Update pos to where the laser finishes this stroke
            path = self.paths[best_idx]
            current_pos = (path.points[0] if best_reverse else path.points[-1]) - self.origin

        jump_time = total_jump_dist / v_jump
        return marking_time + jump_time, sequence
