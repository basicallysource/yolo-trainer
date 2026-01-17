import argparse
import json
import os
import subprocess


def run_cmd(cmd, check=True):
    print(f"  → {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def push(config, remote, remote_path):
    # Create remote directory structure
    run_cmd(["ssh", remote, f"mkdir -p {remote_path}/weights"])

    # Code files to sync
    code_files = [
        "finetune_segmenting.py",
        "config.py",
        "dataset_utils.py",
        "requirements.txt",
    ]

    print("\nSyncing code files...")
    for f in code_files:
        if os.path.exists(f):
            run_cmd(["rsync", "-avz", f, f"{remote}:{remote_path}/"])

    # Sync data directory
    data_path = config["data_path"]
    if os.path.exists(data_path):
        print(f"\nSyncing data from {data_path}...")
        run_cmd([
            "rsync", "-avz", "--progress",
            "--exclude", "__pycache__",
            f"{data_path}/", f"{remote}:{remote_path}/data/"
        ])
    else:
        print(f"Warning: data_path '{data_path}' not found locally")

    # Sync yaml file
    yaml_path = config["data_yaml_path"]
    if os.path.exists(yaml_path):
        print(f"\nSyncing yaml from {yaml_path}...")
        run_cmd(["rsync", "-avz", yaml_path, f"{remote}:{remote_path}/"])
    else:
        print(f"Warning: data_yaml_path '{yaml_path}' not found locally")

    # Sync weights if they exist
    weights_dir = "weights"
    if os.path.exists(weights_dir):
        print(f"\nSyncing base model weights...")
        run_cmd([
            "rsync", "-avz", "--progress",
            f"{weights_dir}/", f"{remote}:{remote_path}/weights/"
        ])

    # Generate remote config with adjusted paths
    remote_config = config.copy()
    remote_config["data_path"] = "data"
    remote_config["data_yaml_path"] = os.path.basename(yaml_path)
    remote_config["checkpoints_dir"] = "checkpoints"

    remote_config_local = ".remote_config.json"
    with open(remote_config_local, "w") as f:
        json.dump(remote_config, f, indent=4)
    run_cmd(["rsync", "-avz", remote_config_local, f"{remote}:{remote_path}/config.json"])
    os.remove(remote_config_local)

    print(f"\n{'='*50}")
    print("Sync complete! To train on remote:")
    print(f"  ssh {remote}")
    print(f"  cd {remote_path}")
    print(f"  python finetune_segmenting.py config.json")
    print(f"{'='*50}")


def pull(config, remote, remote_path):
    checkpoints_dir = config.get("checkpoints_dir", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"\nPulling checkpoints from remote...")
    run_cmd([
        "rsync", "-avz", "--progress",
        f"{remote}:{remote_path}/checkpoints/", f"{checkpoints_dir}/"
    ])

    print(f"\nCheckpoints synced to {checkpoints_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Sync training files to/from remote GPU")
    parser.add_argument("action", choices=["push", "pull"], help="push to remote or pull checkpoints back")
    parser.add_argument("config_path", type=str, help="Path to JSON config file")
    parser.add_argument("--remote", type=str, required=True, help="Remote host (e.g. user@hostname)")
    parser.add_argument("--remote-path", type=str, default="~/yolo-trainer", help="Remote base path (default: ~/yolo-trainer)")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    if args.action == "push":
        push(config, args.remote, args.remote_path)
    else:
        pull(config, args.remote, args.remote_path)


if __name__ == "__main__":
    main()
