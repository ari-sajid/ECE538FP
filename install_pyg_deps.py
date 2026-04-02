"""
install_pyg_deps.py
-------------------
Installs the PyTorch-Geometric C++ extension wheels that match the exact
PyTorch version and CUDA flavour already present in your environment.

Works on Windows, macOS, and Linux.  Run once after  pip install -r requirements.txt:

    python install_pyg_deps.py

The script:
  1. Reads the installed PyTorch version (e.g. "2.2.2").
  2. Detects whether CUDA is available and which version (e.g. "cu121" / "cpu").
  3. Fetches the correct pre-built wheel URL from  https://data.pyg.org/whl/
  4. Calls pip to install all four extension packages in one shot.
"""

import subprocess
import sys


def get_torch_tag() -> str:
    """Return a wheel tag like  torch-2.2.2+cu121  or  torch-2.2.2+cpu."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed.  Run  pip install torch  first.")
        sys.exit(1)

    version = torch.__version__.split("+")[0]   # strip any local suffix

    if torch.cuda.is_available():
        # torch.version.cuda → e.g. "12.1", "11.8"
        cuda_raw = (torch.version.cuda or "").replace(".", "")
        cuda_tag = f"cu{cuda_raw}" if cuda_raw else "cpu"
    else:
        cuda_tag = "cpu"

    tag = f"torch-{version}+{cuda_tag}"
    return tag, version, cuda_tag


def main():
    tag, version, cuda_tag = get_torch_tag()
    wheel_url = f"https://data.pyg.org/whl/{tag}.html"

    packages = [
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
    ]

    print(f"Detected PyTorch {version} with CUDA flavour: {cuda_tag}")
    print(f"Wheel index : {wheel_url}")
    print(f"Installing  : {', '.join(packages)}\n")

    cmd = [
        sys.executable, "-m", "pip", "install",
        *packages,
        "-f", wheel_url,
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\nAll PyG C++ extensions installed successfully.")
        print("You can now run:  python -m src.train")
    else:
        print("\nInstallation failed.")
        print("If pre-built wheels are not available for your platform/version,")
        print("torch-geometric will still work using its pure-Python fallbacks.")
        print("Training will be slower but functionally identical.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
