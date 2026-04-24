#!/usr/bin/env python
"""
Unified training wrapper for 3D Fracture Classification using PointNeXt.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train 3D Fracture Classification.")
    parser.add_argument("--dataset", choices=["fb", "bb"], required=True, help="fb=Fantastic Breaks, bb=Breaking Bad")
    parser.add_argument("--model", choices=["pointnext-b", "pointnext-s"], default="pointnext-b")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable WandB logging")
    parser.add_argument("--background", action="store_true", help="Run as background process")
    
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parents[1]
    pointnext_dir = project_root / "src" / "pointnext"
    
    dataset_name = "fantasticbreaks" if args.dataset == "fb" else "breakingbad"
    cfg_path = f"cfgs/{dataset_name}/{args.model}.yaml"
    
    # Build command
    cmd = [
        sys.executable,
        "examples/classification/main.py",
        "--cfg", cfg_path,
        f"wandb.use_wandb={str(args.wandb).lower()}"
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = f"{pointnext_dir}:{env.get('PYTHONPATH', '')}"

    print(f"Launching training for {dataset_name} on GPU {args.gpu}...")
    print(f"Command: {' '.join(cmd)}")
    
    if args.background:
        # Launch in background using nohup-like behavior
        log_file = project_root / f"train_{dataset_name}_{args.model}.log"
        with open(log_file, "w") as f:
            subprocess.Popen(
                cmd,
                cwd=pointnext_dir,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp
            )
        print(f"Training started in background. Logs: {log_file}")
    else:
        # Run in foreground
        subprocess.run(cmd, cwd=pointnext_dir, env=env)

if __name__ == "__main__":
    main()
