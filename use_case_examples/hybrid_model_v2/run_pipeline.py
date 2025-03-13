#!/usr/bin/env python
import argparse
import os
import signal
import subprocess
import uuid
import time

def run_server():
    # Open the log file for writing.
    log_file = open("server.log", "w")

    # Launch the server with preexec_fn=os.setsid which creates a new process group.
    process = subprocess.Popen(
        ["python", "serve_model.py", "--port", "8000", "--path-to-models", "./compiled_models"],
        stdout=log_file,
        stderr=log_file,
        preexec_fn=os.setsid  # This ensures the subprocess becomes the leader of a new process group.
    )
    return process

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline script to compile the model, run inference, and manage the server."
    )
    parser.add_argument(
        "--module-names", "-M", type=str, required=True,
        help="Comma-separated list of module names."
    )
    parser.add_argument(
        "--num-images", "-n", type=int, required=True,
        help="Number of images to run inference on."
    )
    parser.add_argument(
        "--benchmark-name", "-b", type=str, required=True,
        help="Name of the benchmark."
    )
    parser.add_argument(
        "--snapshot", "-s", type=str, required=True,
        help="Path to snapshot weight file"
    )
    parser.add_argument(
        "--fhe-mode", "-f", type=str, choices=["disable", "remote", "simulate", "calibrate", "execute"],
        default="remote", help="Hybrid FHE mode (disable, remote, simulate, calibrate, execute)"
    )
    args = parser.parse_args()

    # Generate a unique model name using a random UUID.
    model_name = f"model_{uuid.uuid4().hex}"
    print(f"Generated model name: {model_name}")

    # Compile the model with compile_hybrid.py.
    compile_cmd = [
        "python",
        "compile_hybrid.py",
        "-m", model_name,
        "-M", args.module_names,
        "-s", args.snapshot
    ]
    print("Running compile_hybrid.py with command:")
    print(" ".join(compile_cmd))
    subprocess.run(compile_cmd, check=True)

    # Start the server in the background.
    server_process = None
    try:
        print("Starting server in the background...")
        server_process = run_server()

        # Wait a few seconds to allow the server to initialize.
        time.sleep(5)

        # Call infer.py (benchmark.py) with the generated model name and other parameters.
        # We remove the extra module list that was causing the error.
        infer_cmd = [
            "python",
            "benchmark.py",
            "-b", args.benchmark_name,
            "-n", str(args.num_images),
            "-m", model_name,
            "-M", args.module_names,
            "-s", args.snapshot,
            "-f", args.fhe_mode
        ]
        print("Running infer.py with command:")
        print(" ".join(infer_cmd))
        subprocess.run(infer_cmd, check=True)
    finally:
        if server_process is not None:
            print("Terminating server process group...")
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            server_process.wait()
            print("Server terminated.")

if __name__ == "__main__":
    main()