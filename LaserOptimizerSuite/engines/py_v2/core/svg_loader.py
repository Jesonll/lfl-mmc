import xml.etree.ElementTree as ET
import numpy as np
import re
from svg.path import parse_path, Line, Move, CubicBezier, QuadraticBezier, Arc
from dataclasses import dataclass
from typing import List

@dataclass
class LaserPath:
    id: int
    points: np.ndarray
    length: float
    @property
    def start(self): return self.points[0]
    @property
    def end(self): return self.points[-1]

class SVGLoader:
    @staticmethod
    def load_svg(filepath: str, scale: float = 0.001, sampling_resolution: float = 50.0) -> List[LaserPath]:
        """
        sampling_resolution: Points per unit length. 
        Increased default to 50.0 to handle small coordinate SVGs (1x1) smoothly.
        """
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except Exception as e:
            print(f"[Error] Failed to parse XML: {e}")
            return []

        paths = []
        path_counter = 0

        def add_path(pts_list):
            nonlocal path_counter
            if len(pts_list) < 2: return
            pts_arr = np.array(pts_list) * scale
            length = np.sum(np.sqrt(np.sum(np.diff(pts_arr, axis=0)**2, axis=1)))
            if length < 1e-9: return
            paths.append(LaserPath(path_counter, pts_arr, length))
            path_counter += 1

        def discretize_segment(segment):
            """Intelligently discretize a curve segment"""
            length = segment.length()
            if length < 1e-6: return []
            
            # Ensure at least 10 points for any curve, or use resolution
            # If it's a straight line, we only need 2 points
            if isinstance(segment, Line):
                n_steps = 2
            else:
                # Dynamic step count: Min 10, or based on resolution
                n_steps = max(10, int(length * sampling_resolution))
                # Cap at 200 to prevent explosion on huge lines
                n_steps = min(200, n_steps)

            pts = []
            for t in np.linspace(0, 1, n_steps):
                p = segment.point(t)
                pts.append([p.real, p.imag])
            return pts

        def traverse(element):
            tag = element.tag.split('}')[-1].lower()
            
            # --- PATH ---
            if tag == 'path':
                d = element.get('d')
                if d:
                    try:
                        parsed = parse_path(d)
                        curr_pts = []
                        for seg in parsed:
                            if isinstance(seg, Move):
                                add_path(curr_pts)
                                curr_pts = []
                            else:
                                new_pts = discretize_segment(seg)
                                if not new_pts: continue
                                # Avoid duplicating connection points
                                if curr_pts and new_pts:
                                    # If start of new matches end of old, skip first
                                    if np.linalg.norm(np.array(curr_pts[-1]) - np.array(new_pts[0])) < 1e-6:
                                        curr_pts.extend(new_pts[1:])
                                    else:
                                        curr_pts.extend(new_pts)
                                else:
                                    curr_pts.extend(new_pts)
                        add_path(curr_pts)
                    except Exception as e:
                        # Fallback for simple paths if svg.path fails
                        pass

            # --- CIRCLE/ELLIPSE (Converted to Path logic) ---
            elif tag in ['circle', 'ellipse']:
                cx = float(element.get('cx', 0))
                cy = float(element.get('cy', 0))
                rx = float(element.get('r', 0)) if tag == 'circle' else float(element.get('rx', 0))
                ry = float(element.get('r', 0)) if tag == 'circle' else float(element.get('ry', 0))
                
                # High resolution for circles (60 points min)
                steps = max(60, int(2 * np.pi * max(rx, ry) * sampling_resolution))
                theta = np.linspace(0, 2*np.pi, steps)
                pts = np.column_stack([cx + rx*np.cos(theta), cy + ry*np.sin(theta)])
                add_path(pts.tolist())

            # --- RECT/LINE/POLY ---
            elif tag == 'rect':
                x, y = float(element.get('x',0)), float(element.get('y',0))
                w, h = float(element.get('width',0)), float(element.get('height',0))
                add_path([[x,y], [x+w,y], [x+w,y+h], [x,y+h], [x,y]])
            elif tag == 'line':
                x1, y1 = float(element.get('x1',0)), float(element.get('y1',0))
                x2, y2 = float(element.get('x2',0)), float(element.get('y2',0))
                add_path([[x1,y1], [x2,y2]])
            elif tag in ['polyline', 'polygon']:
                pts_str = element.get('points')
                if pts_str:
                    clean = re.sub(r'[,\s]+', ' ', pts_str).strip()
                    nums = list(map(float, clean.split()))
                    pts = np.array(nums).reshape(-1, 2)
                    if tag == 'polygon': pts = np.vstack([pts, pts[0]])
                    add_path(pts.tolist())

            for child in element: traverse(child)

        traverse(root)
        
        if not paths:
            print(f"[Warning] No geometry loaded from {filepath}. Returning Mock Data.")
            return SVGLoader.generate_mock_data()
        
        print(f"[Input] Loaded {len(paths)} paths from {filepath}")
        return paths

    @staticmethod
    def generate_mock_data():
        """High-Res Mock Data"""
        print("[Input] Generating High-Res Mock Data (Spirals)...")
        paths = []
        np.random.seed(42)
        for i in range(50):
            cx, cy = np.random.rand(2) * 0.8 + 0.1
            # 100 steps = Smooth curves
            theta = np.linspace(0, 4*np.pi, 100) 
            r = np.linspace(0.01, 0.05, 100)
            pts = np.column_stack([cx + r*np.cos(theta), cy + r*np.sin(theta)])
            length = np.sum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1)))
            paths.append(LaserPath(i, pts, length))
        return paths
