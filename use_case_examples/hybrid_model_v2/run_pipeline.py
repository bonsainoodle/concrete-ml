#!/usr/bin/env python
import argparse
import subprocess
import uuid
import time
import multiprocessing

def run_server():
    subprocess.run(["sh", "serve.sh"], check=True)

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
    args = parser.parse_args()

    # Generate a unique model name using a random UUID.
    model_name = f"model_{uuid.uuid4().hex}"
    print(f"Generated model name: {model_name}")

    # Compile the model with compile_hybrid.py.
    compile_cmd = [
        "python",
        "compile_hybrid.py",
        "--model-name", model_name,
        "--module-names", args.module_names,
        "--snapshot", args.snapshot
    ]
    print("Running compile_hybrid.py with command:")
    print(" ".join(compile_cmd))
    subprocess.run(compile_cmd, check=True)

    # Start the server in the background using ./serve.sh.
    print("Starting server with ./serve.sh in the background...")

    # Use multiprocessing to start the server process instead of threading.
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()
    
    # Wait a few seconds to allow the server to initialize.
    time.sleep(5)  # Adjust the duration as needed for your environment

    # Prepare the module list for infer.py (if needed by the script).
    module_list = [m.strip() for m in args.module_names.split(",") if m.strip()]

    # Call infer.py (benchmark.py) with the generated model name and other parameters.
    infer_cmd = [
        "python",
        "benchmark.py",
        "-b", args.benchmark_name,
        "-n", str(args.num_images),
        "-m", model_name,
        "-M", args.module_names,
        "-s", args.snapshot
    ] + module_list
    print("Running infer.py with command:")
    print(" ".join(infer_cmd))
    subprocess.run(infer_cmd, check=True)

    # Ensure the server process is properly terminated after the benchmark is complete.
    server_process.join(timeout=3)
    if server_process.is_alive():
        print("Server process did not terminate in time, attempting to kill it.")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            print("Failed to terminate server process in time.")
        else:
            print("Successfully terminated server process.")

if __name__ == "__main__":
    main()
