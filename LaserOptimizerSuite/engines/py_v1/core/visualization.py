import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import CheckButtons
import numpy as np
from scipy.spatial.distance import pdist, squareform
from .lower_layer import LowerLayerSolver

class InteractiveVisualizer:
    def __init__(self, all_paths, stay_points, allocations, history, config):
        self.all_paths = all_paths
        self.stay_points = stay_points
        self.allocations = allocations
        self.history = history
        self.config = config
        
        # Store graphic artists to toggle visibility later
        self.artists = {
            'Platform Path': [],
            'Station Boxes': [],
            'Station Labels': [],
            'Laser Cuts': [],
            'Jump Paths': [],
            'Jump Points (ON/OFF)': [] # Scatter plots
        }

    def show(self):
        print("[Vis] Generating Interactive Dashboard...")
        # Adjust figure size to make room for buttons
        fig = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(right=0.85) # Make room on the right for buttons
        
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Dual-Mode Laser Simulation (Interactive)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Global X Position (m)")
        ax.set_ylabel("Global Y Position (m)")
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.3)

        # --- DRAWING LOGIC ---
        self._draw_scene(ax)

        # --- WIDGETS ---
        # Define the check buttons
        rax = plt.axes([0.87, 0.4, 0.12, 0.3]) # [left, bottom, width, height]
        rax.set_title("Layer Control")
        
        labels = list(self.artists.keys())
        # Default visibility: All ON except maybe text if crowded
        actives = [True] * len(labels) 
        
        check = CheckButtons(rax, labels, actives)

        # Callback function
        def func(label):
            # Get the list of artists associated with this label
            artist_list = self.artists[label]
            for artist in artist_list:
                # Toggle visibility
                artist.set_visible(not artist.get_visible())
            plt.draw()

        check.on_clicked(func)

        # Add a static legend for context (colors)
        self._add_static_legend(ax)

        plt.show()

    def _draw_scene(self, ax):
        field_size = self.config['machine']['galvo_field_size']
        used_indices = np.unique(self.allocations)
        active_sps = self.stay_points[used_indices]

        # 1. Platform Path (TSP Order)
        if len(active_sps) > 0:
            pts_with_home = np.vstack(([0,0], active_sps))
            dists = squareform(pdist(pts_with_home))
            visited = [False] * len(pts_with_home)
            curr = 0
            visited[0] = True
            tsp_order = [] 
            
            for _ in range(len(pts_with_home)-1):
                dists[curr][visited] = np.inf
                next_node = np.argmin(dists[curr])
                tsp_order.append(next_node)
                visited[next_node] = True
                curr = next_node
            
            # Draw Home
            ax.plot(0, 0, 'kD', markersize=10, zorder=5, label='Home')

            prev_pos = np.array([0.0, 0.0])
            for i, idx_in_home_list in enumerate(tsp_order):
                real_sp_idx = used_indices[idx_in_home_list - 1]
                sp_pos = self.stay_points[real_sp_idx]

                # -- Platform Line --
                line, = ax.plot([prev_pos[0], sp_pos[0]], [prev_pos[1], sp_pos[1]], 
                                'b--', linewidth=1.5, alpha=0.6)
                self.artists['Platform Path'].append(line)

                # -- Station Box --
                rect_origin = sp_pos - field_size/2
                rect = patches.Rectangle(rect_origin, field_size, field_size, 
                                         lw=1, ec='blue', fc='none', alpha=0.5, linestyle='-')
                ax.add_patch(rect)
                self.artists['Station Boxes'].append(rect)

                # -- Station Label --
                txt = ax.text(sp_pos[0], sp_pos[1] + field_size/2 + 0.005, 
                              f"St {real_sp_idx}\n#{i+1}", 
                              color='blue', ha='center', fontsize=8, fontweight='bold')
                self.artists['Station Labels'].append(txt)
                
                # -- Station Center Marker --
                center_mark, = ax.plot(sp_pos[0], sp_pos[1], 'bx', markersize=6)
                self.artists['Station Boxes'].append(center_mark)

                # -- Galvo Processing --
                cluster_indices = np.where(self.allocations == real_sp_idx)[0]
                cluster_paths = [self.all_paths[k] for k in cluster_indices]
                
                solver = LowerLayerSolver(cluster_paths, sp_pos, self.config)
                _, sequence = solver.solve()
                
                curr_head = sp_pos
                for p_idx, rev in sequence:
                    path = cluster_paths[p_idx]
                    start = path.points[-1] if rev else path.points[0]
                    end = path.points[0] if rev else path.points[-1]

                    # 1. Jump Path (Gray)
                    j_line, = ax.plot([curr_head[0], start[0]], [curr_head[1], start[1]], 
                                      color='#555555', linestyle=':', lw=1.0, alpha=0.7)
                    self.artists['Jump Paths'].append(j_line)

                    # 2. Jump Points (Start=Green, End=Red)
                    # We use scatter for points. 
                    # Start Point (Laser ON)
                    sc1 = ax.scatter(start[0], start[1], s=20, c='green', zorder=10, edgecolors='none')
                    # End Point (Laser OFF)
                    sc2 = ax.scatter(end[0], end[1], s=20, c='red', zorder=10, edgecolors='none')
                    self.artists['Jump Points (ON/OFF)'].append(sc1)
                    self.artists['Jump Points (ON/OFF)'].append(sc2)

                    # 3. Cut Path (Red)
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
            Line2D([0], [0], color='r', lw=1.5, label='Laser Cut (ON)'),
            Line2D([0], [0], color='#555555', lw=1, ls=':', label='Jump (OFF)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='g', label='Start Cut'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='r', label='End Cut'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

# Wrapper function to keep main.py compatibility
def plot_results(all_paths, stay_points, allocations, history, config):
    vis = InteractiveVisualizer(all_paths, stay_points, allocations, history, config)
    vis.show()