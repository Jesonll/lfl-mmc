import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
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
        # Factor 2.0 to ensure plenty of stations for overlap
        raw_K = int((area_bb / (field_size**2)) * 2.0) + 1
        
        # --- SAFETY CAP ---
        if raw_K > 500:
            print(f"[Warning] Calculated need for {raw_K} stations! This implies a Scale Mismatch.")
            print(f"          Pattern Area: {area_bb:.2f} m2 vs Field Area: {field_size**2:.4f} m2")
            print(f"          Clamping K to 100. PLEASE CHECK 'svg_scale_factor' IN CONFIG.")
            self.K = 100
        else:
            self.K = raw_K
        print(f"[Upper] Estimated {self.K} stay points needed.")

    def _solve_platform_tsp(self, points):
        if len(points) == 0: return 0.0
        pts_with_home = np.vstack(([0,0], points))
        dists = squareform(pdist(pts_with_home))
        n = len(pts_with_home)
        visited = [False] * n
        curr = 0
        visited[0] = True
        total_dist = 0.0
        for _ in range(n-1):
            mask = np.array(visited)
            row = dists[curr].copy()
            row[mask] = np.inf
            nearest = np.argmin(row)
            total_dist += row[nearest]
            curr = nearest
            visited[curr] = True
        return total_dist

    def evaluate(self, particles):
        scores = []
        field_size = self.config['machine']['galvo_field_size']
        half_l = field_size / 2.0
        penalty_weight = self.config['optimization']['time_penalty_weight']
        
        for p_flat in particles:
            stay_points = p_flat.reshape((-1, 2))
            if np.any(np.isnan(stay_points)):
                scores.append(float('inf'))
                continue

            # 1. Assign to nearest
            dists = cdist(self.path_centers, stay_points)
            labels = np.argmin(dists, axis=1)
            
            total_time = 0.0
            penalty = 0.0
            active_indices = []

            for i in range(len(stay_points)):
                cluster_idxs = np.where(labels == i)[0]
                if len(cluster_idxs) == 0: continue
                
                sp = stay_points[i]
                cluster_paths = [self.all_paths[ix] for ix in cluster_idxs]
                
                # --- HARD CONSTRAINT CHECK ---
                # Calculate max deviation from the Stay Point
                all_pts_list = [path.points for path in cluster_paths]
                if all_pts_list:
                    all_pts = np.vstack(all_pts_list)
                    # Vector from Stay Point to Pattern Points
                    local_pts = all_pts - sp
                    
                    # Check X and Y bounds independently
                    # It must lie within -L/2 to +L/2
                    max_abs_dev = np.max(np.abs(local_pts))
                    
                    if max_abs_dev > half_l:
                        # Physical Violation: Laser cannot reach here
                        violation_dist = max_abs_dev - half_l
                        # Massive penalty per meter of violation
                        penalty += violation_dist * penalty_weight
                    else:
                        # Only if valid, calculate time
                        active_indices.append(i)
                        solver = LowerLayerSolver(cluster_paths, sp, self.config)
                        t_galvo, _ = solver.solve()
                        total_time += t_galvo

            if active_indices:
                active_sps = stay_points[active_indices]
                plat_dist = self._solve_platform_tsp(active_sps)
                total_time += plat_dist / self.config['machine']['platform_speed']

            scores.append(total_time + penalty)
        return np.array(scores)

    def optimize(self):
        # ... (Same PSO logic as before, omitting to save space) ...
        # ... The only change is calling self._refine_centering at the end ...
        
        # [Copy/Paste the PSO loop from previous answer here if overwriting]
        # For brevity, I assume the optimize loop logic exists.
        # Below is the Refine Centering logic which is critical:

        dim = self.K * 2
        pop_size = self.config['optimization']['pso_particles']
        iterations = self.config['optimization']['pso_iterations']
        
        # Init
        unique_centers = np.unique(self.path_centers, axis=0)
        if len(unique_centers) >= self.K:
            try:
                centroids, _ = kmeans2(self.path_centers, self.K, minit='points')
                p0 = centroids.flatten()
            except:
                p0 = np.random.rand(dim) * self.config['process']['canvas_size_m']
        else:
            p0 = np.random.rand(dim) * self.config['process']['canvas_size_m']

        X = np.random.rand(pop_size, dim) * self.config['process']['canvas_size_m']
        X[0] = p0
        V = np.random.randn(pop_size, dim) * 0.1
        
        pbest_X = X.copy()
        pbest_scores = self.evaluate(X)
        gbest_idx = np.argmin(pbest_scores)
        gbest_X = pbest_X[gbest_idx]
        gbest_score = pbest_scores[gbest_idx]
        history = []
        
        print(f"[Upper] PSO Start...")
        for i in range(iterations):
            r1, r2 = np.random.rand(2)
            w = 0.7 - (0.4 * i / iterations)
            c1, c2 = 1.4, 1.4
            V = w*V + c1*r1*(pbest_X - X) + c2*r2*(gbest_X - X)
            X = X + V
            X = np.clip(X, 0, self.config['process']['canvas_size_m'])
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

        # Final Refinement: Move Stay Points to the EXACT center of their clusters
        # This maximizes the margin from the boundary.
        final_points = gbest_X.reshape((-1, 2))
        refined_points, labels = self._refine_centering(final_points)
        
        return refined_points, labels, gbest_score, history

    def _refine_centering(self, stay_points):
        dists = cdist(self.path_centers, stay_points)
        labels = np.argmin(dists, axis=1)
        refined = stay_points.copy()
        used_indices = np.unique(labels)
        
        for idx in used_indices:
            cluster_idxs = np.where(labels == idx)[0]
            cluster_paths = [self.all_paths[i] for i in cluster_idxs]
            
            # Get Bounding Box of Cluster
            all_pts = np.vstack([p.points for p in cluster_paths])
            min_xy = np.min(all_pts, axis=0)
            max_xy = np.max(all_pts, axis=0)
            
            # Set Stay Point to Geometric Center
            center = (min_xy + max_xy) / 2.0
            refined[idx] = center
            
        return refined, labels