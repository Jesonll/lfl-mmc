#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import shutil
import json
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINES_DIR = os.path.join(ROOT_DIR, "engines")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
INPUT_DIR = os.path.join(ROOT_DIR, "inputs")

def run_command(cmd, cwd=None, env=None):
    try:
        subprocess.check_call(cmd, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[!] Error. Code: {e.returncode}")
        sys.exit(1)

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

def print_evaluation(metrics_path):
    if not os.path.exists(metrics_path): return
    with open(metrics_path, 'r') as f: data = json.load(f)
    t_compute = data.get("time_compute_sec", 0.0)
    t_machine = data.get("time_total_machine_sec", 0.0)
    res_mm = data.get("geo_resolution_mm", 0.0)
    
    print("\n" + "="*50)
    print(f"   FINAL EVALUATION REPORT")
    print("="*50)
    print(f"1. ACCURACY")
    print(f"   - Avg Segment Len:    {res_mm:.4f} mm")
    print("-" * 50)
    print(f"2. TIME")
    print(f"   - Computation:        {t_compute:.4f} s")
    print(f"   - Machining:          {t_machine:.4f} s")
    print("="*50 + "\n")

def install_system(args):
    print("=== INSTALLING... ===")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    build_dir = os.path.join(ENGINES_DIR, "cpp_v3", "build")
    if os.path.exists(build_dir) and args.clean: shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)
    run_command(["cmake", ".."], cwd=build_dir)
    run_command(["make", f"-j{os.cpu_count() or 1}"], cwd=build_dir)
    print("[✓] Ready.")

def run_job(args):
    ensure_dirs()
    input_path = os.path.abspath(args.input) if os.path.exists(args.input) else os.path.join(INPUT_DIR, args.input)
    if not os.path.exists(input_path): print("[!] Input not found."); sys.exit(1)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_gcode = os.path.join(OUTPUT_DIR, f"{base_name}_{args.engine}.gcode")
    out_html = os.path.join(OUTPUT_DIR, f"{base_name}_{args.engine}.html")
    
    print(f"=== ENGINE: {args.engine.upper()} ===")
    
    cmd = []
    cwd = None

    if args.engine == "cpp":
        exe = os.path.join(ENGINES_DIR, "cpp_v3", "build", "laser_planner")
        if not os.path.exists(exe): print("[!] Run install first."); sys.exit(1)
        
        cmd = [
            exe, input_path,
            "--gcode_output", out_gcode,
            "--output", out_html,
            "--segment_len", str(args.segment_len),
            "--platform_speed", str(args.platform_speed),
            "--galvo_mark_speed", str(args.mark_speed),
            "--galvo_jump_speed", str(args.jump_speed),
            "--galvo_field_size", str(args.field_size),
        ]
        if args.width: cmd.extend(["--normalize", str(args.width), str(args.height)])
        else: cmd.extend(["--svg_scale_factor", str(args.scale)])

    elif args.engine in ["v1", "v2"]:
        # (Python logic same as before, simplified for brevity)
        engine_dir = os.path.join(ENGINES_DIR, f"py_{args.engine}")
        cwd = engine_dir
        cmd = [sys.executable, "main.py", "--input", input_path, "--output", out_gcode, "--segment_len", str(args.segment_len)]
        if args.width: cmd.extend(["--target_size", str(args.width), str(args.height)])
        if args.tile > 1: cmd.extend(["--tile", str(args.tile)])

    run_command(cmd, cwd=cwd)
    print_evaluation(out_gcode + ".json")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_inst = subparsers.add_parser("install")
    p_inst.add_argument("--clean", action="store_true")
    p_run = subparsers.add_parser("run")
    p_run.add_argument("input")
    p_run.add_argument("--engine", choices=["cpp", "v1", "v2"], default="cpp")
    p_run.add_argument("--width", type=float)
    p_run.add_argument("--height", type=float)
    p_run.add_argument("--scale", type=float, default=1.0)
    p_run.add_argument("--tile", type=int, default=1)
    p_run.add_argument("--segment_len", type=float, default=0.05)
    p_run.add_argument("--platform_speed", type=float, default=0.25)
    p_run.add_argument("--mark_speed", type=float, default=2.5)
    p_run.add_argument("--jump_speed", type=float, default=5.0)
    p_run.add_argument("--field_size", type=float, default=0.2)
    args = parser.parse_args()
    if args.command == "install": install_system(args)
    elif args.command == "run": run_job(args)

if __name__ == "__main__":
    main()