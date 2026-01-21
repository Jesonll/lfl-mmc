#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil
import json
import platform

# --- CONFIGURATION ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINES_DIR = os.path.join(ROOT_DIR, "engines")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
INPUT_DIR = os.path.join(ROOT_DIR, "inputs")

# --- UTILS ---
def run_command(cmd, cwd=None, env=None):
    """Runs a shell command and streams output"""
    print(f"[$] {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[!] Error executing command. Return code: {e.returncode}")
        sys.exit(1)

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

# --- INSTALLATION HANDLER ---
def install_system(args):
    print("=== INSTALLING LASER OPTIMIZER SUITE ===")
    
    # 1. Python Dependencies
    print("\n[1/2] Installing Python Dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # 2. C++ Compilation
    print("\n[2/2] Compiling C++ Engine (Cpp-v3)...")
    cpp_dir = os.path.join(ENGINES_DIR, "cpp_v3")
    build_dir = os.path.join(cpp_dir, "build")
    
    if os.path.exists(build_dir) and args.clean:
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    # Detect CMake
    cmake_cmd = "cmake"
    if shutil.which("cmake3"): cmake_cmd = "cmake3"
    
    # Configure & Build
    run_command([cmake_cmd, ".."], cwd=build_dir)
    
    # Multi-core build
    cpu_count = os.cpu_count() or 1
    run_command(["make", f"-j{cpu_count}"], cwd=build_dir)
    
    print("\n[✓] Installation Complete.")

# --- EXECUTION HANDLER ---
def run_job(args):
    ensure_dirs()
    
    # 1. Resolve Input File
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(INPUT_DIR, input_path)
    
    if not os.path.exists(input_path):
        print(f"[!] Input file not found: {input_path}")
        sys.exit(1)

    # 2. Prepare Output Paths
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_html = os.path.join(OUTPUT_DIR, f"{base_name}_{args.engine}.html")
    out_gcode = os.path.join(OUTPUT_DIR, f"{base_name}_{args.engine}.gcode")

    print(f"=== LAUNCHING ENGINE: {args.engine.upper()} ===")
    
    # --- STRATEGY: C++ (CPP-V3) ---
    if args.engine == "cpp":
        executable = os.path.join(ENGINES_DIR, "cpp_v3", "build", "laser_planner")
        if not os.path.exists(executable):
            print("[!] C++ executable not found. Run 'python manager.py install' first.")
            sys.exit(1)

        cmd = [
            executable,
            input_path,
            "--platform_speed", str(args.platform_speed),
            "--galvo_mark_speed", str(args.mark_speed),
            "--galvo_jump_speed", str(args.jump_speed),
            "--galvo_field_size", str(args.field_size),
            "--output", out_html,
            "--gcode_output", out_gcode
        ]
        
        if args.width and args.height:
            cmd.extend(["--normalize", str(args.width), str(args.height)])
        else:
            cmd.extend(["--svg_scale_factor", str(args.scale)])

        run_command(cmd)

    # --- STRATEGY: PYTHON (V1 / V2) ---
    elif args.engine in ["v1", "v2"]:
        engine_dir = os.path.join(ENGINES_DIR, f"py_{args.engine}")
        main_script = os.path.join(engine_dir, "main.py")
        config_path = os.path.join(engine_dir, "config.json")
        
        # 1. Dynamically Update Config.json based on CLI args
        # (This bridges the gap between CLI and the Python internal config)
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            
            cfg["machine"]["platform_speed"] = args.platform_speed
            cfg["machine"]["galvo_mark_speed"] = args.mark_speed
            cfg["machine"]["galvo_jump_speed"] = args.jump_speed
            cfg["machine"]["galvo_field_size"] = args.field_size
            
            # Handle Normalization via Scale Factor or custom target logic
            # (Assuming Py versions have been updated to support target_size or we rely on scale)
            cfg["process"]["svg_scale_factor"] = args.scale
            # Note: For strict normalization, Py versions need the update I provided in previous answers
            
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=4)
        except Exception as e:
            print(f"[!] Failed to update config.json: {e}")

        # 2. Run Python
        cmd = [
            sys.executable,
            main_script,
            "--input", input_path,
            "--output", out_gcode,
            "--config", config_path
        ]
        
        # Pass normalization flags if the underlying python script supports it (Py-v2 does)
        if args.width and args.height:
            cmd.extend(["--target_size", str(args.width), str(args.height)])
        
        if args.tile > 1:
            cmd.extend(["--tile", str(args.tile)])

        run_command(cmd, cwd=engine_dir)
        
        # Move generated HTML (if any) to output dir
        # (Assuming visualization is saved locally in engine dir)
        # implementation detail: user might need to adjust Py scripts to save to specific path

# --- CLI DEFINITION ---
def main():
    parser = argparse.ArgumentParser(description="Laser Optimizer Suite Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: INSTALL
    p_install = subparsers.add_parser("install", help="Build C++ engine and install deps")
    p_install.add_argument("--clean", action="store_true", help="Clean rebuild")

    # Command: RUN
    p_run = subparsers.add_parser("run", help="Execute an optimization job")
    
    # Required
    p_run.add_argument("input", help="Input SVG filename (in inputs/ folder or absolute path)")
    
    # Engine Selection
    p_run.add_argument("--engine", choices=["cpp", "v1", "v2"], default="cpp", 
                       help="Choose backend: cpp (Fastest), v2 (Co-Evo), v1 (Basic)")

    # Physics Param
    p_run.add_argument("--platform_speed", type=float, default=0.25, help="Stage speed (m/s)")
    p_run.add_argument("--mark_speed", type=float, default=2.5, help="Laser marking speed (m/s)")
    p_run.add_argument("--jump_speed", type=float, default=5.0, help="Galvo jump speed (m/s)")
    p_run.add_argument("--field_size", type=float, default=0.2, help="Scan field size (m)")

    # Geometry Param
    p_run.add_argument("--width", type=float, help="Target Normalize Width (m)")
    p_run.add_argument("--height", type=float, help="Target Normalize Height (m)")
    p_run.add_argument("--scale", type=float, default=1.0, help="Raw scale factor (if not normalizing)")
    p_run.add_argument("--tile", type=int, default=1, help="Tile pattern N x N (Python engines only)")

    args = parser.parse_args()

    if args.command == "install":
        install_system(args)
    elif args.command == "run":
        run_job(args)

if __name__ == "__main__":
    main()
