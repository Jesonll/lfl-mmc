import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2
from .lower_layer import LowerLayerSolver

class UpperLayerOptimizer:
    def __init__(self, all_paths, config):
        self.all_paths = all_paths
        self.config = config
        self.path_centers = np.array([np.mean(p.points, axis=0) for p in all_paths])
        
        # Estimate K (Stay Points)
        field_size = config['machine']['galvo_field_size']
        if len(all_paths) > 0:
            all_pts = np.vstack([p.points for p in all_paths])
            min_xy = np.min(all_pts, axis=0)
            max_xy = np.max(all_pts, axis=0)
            area_bb = np.prod(max_xy - min_xy)
            # Factor 2.0 ensures plenty of overlap potential
            raw_K = int((area_bb / (field_size**2)) * 2.0) + 1
            
            # Safety Cap
            if raw_K > 500:
                print(f"[Upper] Warning: K={raw_K} clamped to 100. Check scaling.")
                self.K = 100
            else:
                self.K = raw_K
        else:
            self.K = 1
            
        print(f"[Upper] Optimization Dimension: {self.K} Stay Points (Coords + Order)")

    def evaluate(self, particles):
        """
        Evaluate particle fitness.
        Particle Shape: (3 * K)
          - [0 : 2K] -> (x, y) coordinates
          - [2K : 3K] -> Permutation Priorities (Random Keys)
        """
        scores = []
        field_size = self.config['machine']['galvo_field_size']
        half_l = field_size / 2.0
        penalty_weight = self.config['optimization']['time_penalty_weight']
        plat_speed = self.config['machine']['platform_speed']
        
        for p_flat in particles:
            # 1. Decode Particle
            # Coordinates
            coords_flat = p_flat[:self.K * 2]
            stay_points = coords_flat.reshape((-1, 2))
            
            # Permutation (Sort priorities to get order)
            priorities = p_flat[self.K * 2:]
            # argsort returns the indices that would sort the array
            # e.g., priorities [0.9, 0.1, 0.5] -> visit_order [1, 2, 0]
            full_visit_order = np.argsort(priorities)
            
            if np.any(np.isnan(stay_points)):
                scores.append(float('inf'))
                continue

            # 2. Assignment (Clustering)
            # Assign every path to the geographically nearest stay point
            dists = cdist(self.path_centers, stay_points)
            allocations = np.argmin(dists, axis=1)
            
            total_time = 0.0
            penalty = 0.0
            
            # Track which stations are actually used
            active_stations_mask = np.zeros(self.K, dtype=bool)
            
            # 3. Lower Layer Cost (Galvo)
            # We iterate 0..K to calculate processing time for each cluster
            for i in range(self.K):
                cluster_idxs = np.where(allocations == i)[0]
                if len(cluster_idxs) == 0: 
                    continue
                
                active_stations_mask[i] = True
                sp = stay_points[i]
                cluster_paths = [self.all_paths[ix] for ix in cluster_idxs]
                
                # Bounds Check
                all_pts_list = [path.points for path in cluster_paths]
                if all_pts_list:
                    all_pts = np.vstack(all_pts_list)
                    local_pts = all_pts - sp
                    max_abs_dev = np.max(np.abs(local_pts))
                    
                    if max_abs_dev > half_l:
                        penalty += (max_abs_dev - half_l) * penalty_weight
                    else:
                        # Only solve path if feasible
                        solver = LowerLayerSolver(cluster_paths, sp, self.config)
                        t_galvo, _ = solver.solve()
                        total_time += t_galvo

            # 4. Upper Layer Cost (Platform Travel)
            # strictly following the Evolved Permutation
            
            # Filter the full order to only include used stations
            # e.g. Order [1, 5, 2], but 5 is empty -> [1, 2]
            active_visit_sequence = [idx for idx in full_visit_order if active_stations_mask[idx]]
            
            if active_visit_sequence:
                current_pos = np.array([0.0, 0.0]) # Start at Home
                travel_dist = 0.0
                
                for sp_idx in active_visit_sequence:
                    next_pos = stay_points[sp_idx]
                    travel_dist += np.linalg.norm(next_pos - current_pos)
                    current_pos = next_pos
                
                total_time += travel_dist / plat_speed

            scores.append(total_time + penalty)
            
        return np.array(scores)

    def optimize(self):
        # 3 dimensions per Stay Point: X, Y, Priority
        dim = self.K * 3
        pop_size = self.config['optimization']['pso_particles']
        iterations = self.config['optimization']['pso_iterations']
        canvas_size = self.config['process']['canvas_size_m']
        
        # --- Initialization ---
        # 1. Coordinates (Smart K-Means Init)
        unique_centers = np.unique(self.path_centers, axis=0)
        if len(unique_centers) >= self.K:
            try:
                centroids, _ = kmeans2(self.path_centers, self.K, minit='points')
                coords_init = centroids.flatten()
            except:
                coords_init = np.random.rand(self.K * 2) * canvas_size
        else:
            coords_init = np.random.rand(self.K * 2) * canvas_size

        # 2. Priorities (Random 0-1)
        priorities_init = np.random.rand(self.K)
        
        # Combine into Particle 0
        p0 = np.concatenate([coords_init, priorities_init])

        # Generate Swarm
        X = np.random.rand(pop_size, dim)
        # Scale coordinates part to canvas size
        X[:, :self.K*2] *= canvas_size
        # Inject smart particle
        X[0] = p0
        
        V = np.random.randn(pop_size, dim) * 0.1
        
        pbest_X = X.copy()
        pbest_scores = self.evaluate(X)
        gbest_idx = np.argmin(pbest_scores)
        gbest_X = pbest_X[gbest_idx]
        gbest_score = pbest_scores[gbest_idx]
        history = []
        
        print(f"[Upper] PSO Start (Co-Evolving Locations & Sequence)...")
        
        for i in range(iterations):
            r1, r2 = np.random.rand(2)
            w = 0.7 - (0.4 * i / iterations)
            c1, c2 = 1.4, 1.4
            
            V = w*V + c1*r1*(pbest_X - X) + c2*r2*(gbest_X - X)
            X = X + V
            
            # Clip Coordinates to Canvas
            X[:, :self.K*2] = np.clip(X[:, :self.K*2], 0, canvas_size)
            # Clip Priorities to [0, 1] (optional, but keeps math clean)
            X[:, self.K*2:] = np.clip(X[:, self.K*2:], 0.0, 1.0)
            
            scores = self.evaluate(X)
            
            mask = scores < pbest_scores
            pbest_X[mask] = X[mask]
            pbest_scores[mask] = scores[mask]
            
            min_curr = np.argmin(scores)
            if scores[min_curr] < gbest_score:
                gbest_score = scores[min_curr]
                gbest_X = X[min_curr]
            
            history.append(gbest_score)
            if (i+1) % 5 == 0:
                print(f"  Iter {i+1} | Cost: {gbest_score:.4f}")

        # --- Final Result Extraction ---
        final_coords_flat = gbest_X[:self.K*2]
        final_priorities = gbest_X[self.K*2:]
        
        # 1. Coordinates
        raw_stay_points = final_coords_flat.reshape((-1, 2))
        
        # 2. Refinement (Centering)
        # Note: We must preserve the permutation logic.
        refined_points, allocations = self._refine_centering(raw_stay_points)
        
        # 3. Determine Final Optimized Sequence
        # Get raw order from priorities
        full_order = np.argsort(final_priorities)
        
        # Filter for only stations that were assigned vectors
        # (Allocations is the map of path->station_index)
        used_indices = np.unique(allocations)
        final_sequence = [idx for idx in full_order if idx in used_indices]
        
        return refined_points, allocations, final_sequence, gbest_score, history

    def _refine_centering(self, stay_points):
        dists = cdist(self.path_centers, stay_points)
        allocations = np.argmin(dists, axis=1)
        refined = stay_points.copy()
        used_indices = np.unique(allocations)
        
        for idx in used_indices:
            cluster_idxs = np.where(allocations == idx)[0]
            cluster_paths = [self.all_paths[i] for i in cluster_idxs]
            
            # Geometric Center
            all_pts = np.vstack([p.points for p in cluster_paths])
            min_xy = np.min(all_pts, axis=0)
            max_xy = np.max(all_pts, axis=0)
            center = (min_xy + max_xy) / 2.0
            refined[idx] = center
            
        return refined, allocations