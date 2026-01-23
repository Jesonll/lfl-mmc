import sys
import re
import numpy as np
from xml.dom import minidom
from svg.path import parse_path, Line, Move, Close

def parse_transform(transform_str):
    """Parses SVG transform string into a 3x3 matrix."""
    mat = np.identity(3)
    if not transform_str: return mat

    # Extract all transforms: translate(..), scale(..), matrix(..)
    transforms = re.findall(r'([a-z]+)\(([^)]+)\)', transform_str)
    
    for name, args in transforms:
        # Clean and split args
        nums = [float(x) for x in re.split(r'[,\s]+', args.strip()) if x]
        t_mat = np.identity(3)
        
        if name == 'translate':
            tx = nums[0]
            ty = nums[1] if len(nums) > 1 else 0
            t_mat[0, 2] = tx
            t_mat[1, 2] = ty
        elif name == 'scale':
            sx = nums[0]
            sy = nums[1] if len(nums) > 1 else sx
            t_mat[0, 0] = sx
            t_mat[1, 1] = sy
        elif name == 'matrix' and len(nums) == 6:
            # SVG matrix: a c e / b d f / 0 0 1
            t_mat = np.array([
                [nums[0], nums[2], nums[4]],
                [nums[1], nums[3], nums[5]],
                [0,       0,       1]
            ])
        
        # Combine (multiply)
        mat = np.dot(mat, t_mat)
        
    return mat

def process_node(node, parent_mat, all_paths):
    """Recursively processes nodes to apply transforms and extract paths."""
    
    # Get local transform
    transform_str = node.getAttribute('transform') if node.hasAttribute('transform') else ''
    local_mat = parse_transform(transform_str)
    
    # Cumulative matrix
    current_mat = np.dot(parent_mat, local_mat)
    
    # 1. Handle <path>
    if node.tagName == 'path':
        d = node.getAttribute('d')
        if d:
            all_paths.append({'d': d, 'matrix': current_mat})
            
    # 2. Handle groups <g> (Recursion)
    for child in node.childNodes:
        if child.nodeType == 1: # Element Node
            process_node(child, current_mat, all_paths)

def flatten(svg_path, out_path):
    print(f"[Flatten] Processing {svg_path}...")
    try:
        doc = minidom.parse(svg_path)
    except Exception as e:
        print(f"[Error] XML Parse failed: {e}")
        return

    root = doc.documentElement
    
    # Extract ALL paths with their baked-in matrices
    extracted_data = []
    process_node(root, np.identity(3), extracted_data)
    
    if not extracted_data:
        print("[Warn] No paths found.")
        return

    # Discretize and Apply Transforms
    final_polys = []
    
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for item in extracted_data:
        d = item['d']
        mat = item['matrix']
        
        try:
            path_obj = parse_path(d)
            current_poly = []
            
            for seg in path_obj:
                length = seg.length()
                if length == 0: continue
                
                # Dynamic resolution: high quality for intermediate file
                # C++ will downsample later if needed.
                steps = max(4, int(length * 2)) # 2 points per pixel approx
                steps = min(steps, 100) # Cap safety
                
                for i in range(steps + 1):
                    t = i / steps
                    pt = seg.point(t)
                    
                    # Apply Transform Matrix
                    vec = np.array([pt.real, pt.imag, 1.0])
                    res = np.dot(mat, vec)
                    
                    x, y = res[0], res[1]
                    
                    # Update Bounds
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y
                    
                    cmd = 'M' if (i==0) else 'L'
                    # Handle breaks
                    if isinstance(seg, Move) and i==0:
                        if current_poly: final_polys.append(current_poly)
                        current_poly = []
                        cmd = 'M'
                        
                    # Optimization: Skip tiny moves
                    if cmd == 'L' and current_poly:
                        last_cmd, lx, ly = current_poly[-1]
                        if abs(lx - x) < 0.001 and abs(ly - y) < 0.001:
                            continue

                    current_poly.append((cmd, x, y))
            
            if current_poly:
                final_polys.append(current_poly)
                
        except Exception as e:
            # Ignore malformed path strings
            pass

    # --- NORMALIZE (Shift to 0,0) ---
    width = max_x - min_x
    height = max_y - min_y
    
    print(f"[Flatten] Bounds: ({min_x:.2f}, {min_y:.2f}) to ({max_x:.2f}, {max_y:.2f})")
    print(f"[Flatten] Shifting to (0,0)... Size: {width:.2f} x {height:.2f}")

    with open(out_path, 'w') as f:
        # Add viewBox for proper previewing in browser/tools
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}">\n')
        
        for poly in final_polys:
            d_parts = []
            for cmd, x, y in poly:
                # SHIFT COORDINATES
                sx = x - min_x
                sy = y - min_y
                d_parts.append(f"{cmd} {sx:.3f} {sy:.3f}")
            
            f.write(f'  <path d="{" ".join(d_parts)}" stroke="black" fill="none" stroke-width="1"/>\n')
            
        f.write('</svg>')
        
    print(f"[Flatten] Saved clean geometry to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python flatten.py in.svg out.svg")
    else:
        flatten(sys.argv[1], sys.argv[2])