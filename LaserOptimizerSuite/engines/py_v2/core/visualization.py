import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import CheckButtons
import numpy as np
from scipy.spatial.distance import pdist, squareform
from .lower_layer import LowerLayerSolver

class InteractiveVisualizer:
    def __init__(self, all_paths, stay_points, allocations, history, config, sequence=None):
        self.all_paths = all_paths
        self.stay_points = stay_points
        self.allocations = allocations
        self.history = history
        self.config = config
        self.sequence = sequence # <--- Store the optimized sequence
        
        self.artists = {
            'Platform Path': [],
            'Station Boxes': [],
            'Station Labels': [],
            'Laser Cuts': [],
            'Jump Paths': [],
            'Jump Points': []
        }

    def show(self):
        print("[Vis] Generating Interactive Dashboard...")
        fig = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(right=0.85)
        
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Dual-Mode Laser Simulation (Interactive)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Global X Position (m)")
        ax.set_ylabel("Global Y Position (m)")
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.3)

        self._draw_scene(ax)

        rax = plt.axes([0.87, 0.4, 0.12, 0.3])
        rax.set_title("Layer Control")
        labels = list(self.artists.keys())
        actives = [True] * len(labels)
        check = CheckButtons(rax, labels, actives)

        def func(label):
            for artist in self.artists[label]:
                artist.set_visible(not artist.get_visible())
            plt.draw()

        check.on_clicked(func)
        self._add_static_legend(ax)
        plt.show()

    def _draw_scene(self, ax):
        field_size = self.config['machine']['galvo_field_size']
        
        # 1. Determine Sequence
        if self.sequence is not None and len(self.sequence) > 0:
            final_order = self.sequence
        else:
            # Fallback Greedy TSP
            used_indices = np.unique(self.allocations)
            if len(used_indices) == 0: return
            active_sps = self.stay_points[used_indices]
            pts_with_home = np.vstack(([0,0], active_sps))
            dists = squareform(pdist(pts_with_home))
            visited = [False] * len(pts_with_home)
            curr = 0
            visited[0] = True
            tsp_indices = []
            for _ in range(len(pts_with_home)-1):
                dists[curr][visited] = np.inf
                next_node = np.argmin(dists[curr])
                tsp_indices.append(next_node)
                visited[next_node] = True
                curr = next_node
            final_order = [used_indices[i-1] for i in tsp_indices]

        # Draw Home
        ax.plot(0, 0, 'kD', markersize=10, zorder=5, label='Home')

        prev_pos = np.array([0.0, 0.0])
        
        for i, real_sp_idx in enumerate(final_order):
            sp_pos = self.stay_points[real_sp_idx]

            # Platform Path
            line, = ax.plot([prev_pos[0], sp_pos[0]], [prev_pos[1], sp_pos[1]], 
                            'b--', linewidth=1.5, alpha=0.6)
            self.artists['Platform Path'].append(line)

            # Station Box
            rect_origin = sp_pos - field_size/2
            rect = patches.Rectangle(rect_origin, field_size, field_size, 
                                     lw=1, ec='blue', fc='none', alpha=0.5)
            ax.add_patch(rect)
            self.artists['Station Boxes'].append(rect)
            
            # Station Center
            c_mark, = ax.plot(sp_pos[0], sp_pos[1], 'bx', markersize=6)
            self.artists['Station Boxes'].append(c_mark)

            # Label
            txt = ax.text(sp_pos[0], sp_pos[1] + field_size/2 + 0.005, 
                          f"St {real_sp_idx}\n#{i+1}", 
                          color='blue', ha='center', fontsize=8, fontweight='bold')
            self.artists['Station Labels'].append(txt)

            # Galvo Content
            cluster_indices = np.where(self.allocations == real_sp_idx)[0]
            cluster_paths = [self.all_paths[k] for k in cluster_indices]
            
            solver = LowerLayerSolver(cluster_paths, sp_pos, self.config)
            _, sequence_micro = solver.solve()
            
            curr_head = sp_pos
            for p_idx, rev in sequence_micro:
                path = cluster_paths[p_idx]
                start = path.points[-1] if rev else path.points[0]
                end = path.points[0] if rev else path.points[-1]

                # Jump
                j_line, = ax.plot([curr_head[0], start[0]], [curr_head[1], start[1]], 
                                  color='#555555', linestyle=':', lw=1.0, alpha=0.7)
                self.artists['Jump Paths'].append(j_line)

                # Jump Points
                sc1 = ax.scatter(start[0], start[1], s=20, c='green', zorder=10, edgecolors='none')
                sc2 = ax.scatter(end[0], end[1], s=20, c='red', zorder=10, edgecolors='none')
                self.artists['Jump Points'].append(sc1)
                self.artists['Jump Points'].append(sc2)

                # Cut
                draw_pts = path.points[::-1] if rev else path.points
                c_line, = ax.plot(draw_pts[:,0], draw_pts[:,1], 'r-', lw=1.2)
                self.artists['Laser Cuts'].append(c_line)

                curr_head = end

            prev_pos = sp_pos

    def _add_static_legend(self, ax):
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='b', lw=1.5, ls='--', label='Platform Travel'),
            patches.Patch(edgecolor='blue', facecolor='none', label='Scanner Field'),
            Line2D([0], [0], color='r', lw=1.5, label='Laser Cut'),
            Line2D([0], [0], color='#555555', lw=1, ls=':', label='Jump'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')

def plot_results(all_paths, stay_points, allocations, history, config, sequence=None):
    vis = InteractiveVisualizer(all_paths, stay_points, allocations, history, config, sequence=sequence)
    vis.show()