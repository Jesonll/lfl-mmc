import sys
import matplotlib.pyplot as plt
import re
import argparse

def parse_line(line):
    # Extracts X, Y from G0/G1 commands
    x_match = re.search(r'X([-\d.]+)', line)
    y_match = re.search(r'Y([-\d.]+)', line)
    
    x = float(x_match.group(1)) if x_match else None
    y = float(y_match.group(1)) if y_match else None
    
    is_platform = "; Platform" in line or "; Move Platform" in line
    # Strict M-code checking
    is_laser_on = line.strip() == "M3" or "M3 " in line
    is_laser_off = line.strip() == "M5" or "M5 " in line
    
    return x, y, is_platform, is_laser_on, is_laser_off

def view_dual_stage_gcode(path):
    print(f"[Viewer] Loading {path}...")
    
    platform_x, platform_y = 0.0, 0.0
    galvo_x, galvo_y = 0.0, 0.0
    
    # Store segments as (x1, y1, x2, y2)
    jumps = [] 
    cuts = []
    platform_moves = []
    
    laser_on = False
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'): continue
            
            x, y, is_platform, m3, m5 = parse_line(line)
            
            if m3: 
                laser_on = True
                continue
            if m5: 
                laser_on = False
                continue

            if is_platform:
                # Platform Movement (Global Shift)
                new_plat_x = x if x is not None else platform_x
                new_plat_y = y if y is not None else platform_y
                
                # Visualizing the Stage movement center-to-center
                platform_moves.append((platform_x, platform_y, new_plat_x, new_plat_y))
                
                platform_x = new_plat_x
                platform_y = new_plat_y
                # Galvo coord system moves with platform, but mirrors stay relative
                
            elif x is not None or y is not None:
                # Galvo Movement (Relative to Platform)
                new_galvo_x = x if x is not None else galvo_x
                new_galvo_y = y if y is not None else galvo_y
                
                # Calculate Absolute World Coordinates for plotting
                abs_start_x = platform_x + galvo_x
                abs_start_y = platform_y + galvo_y
                abs_end_x = platform_x + new_galvo_x
                abs_end_y = platform_y + new_galvo_y
                
                if laser_on:
                    cuts.append((abs_start_x, abs_start_y, abs_end_x, abs_end_y))
                else:
                    jumps.append((abs_start_x, abs_start_y, abs_end_x, abs_end_y))
                
                galvo_x = new_galvo_x
                galvo_y = new_galvo_y

    print(f"[Viewer] Rendering {len(cuts)} cuts...")
    
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_facecolor('#050505')
    fig.patch.set_facecolor('#050505')
    
    # Helper for batch plotting
    def plot_layer(segments, color, alpha, lw, label):
        if not segments: return
        xs, ys = [], []
        # Inserting None creates breaks in the line plot, much faster than plotting individually
        for x1, y1, x2, y2 in segments:
            xs.extend([x1, x2, None])
            ys.extend([y1, y2, None])
        ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw, label=label)

    # 1. Platform (Blue, Bold)
    plot_layer(platform_moves, '#0088ff', 0.8, 1.0, 'Platform')
    
    # 2. Jumps (Gray, Faint) -> This prevents the "Joined" look
    plot_layer(jumps, '#444444', 0.3, 0.2, 'Jump (Off)')
    
    # 3. Cuts (Red, Sharp)
    plot_layer(cuts, '#ff0044', 1.0, 0.4, 'Cut (On)')

    ax.set_aspect('equal')
    # Remove axis clutter
    ax.axis('off')
    
    plt.title(f"G-Code Verification: {len(cuts)} segments", color='white')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 view_gcode_dual.py output.gcode")
    else:
        view_dual_stage_gcode(sys.argv[1])