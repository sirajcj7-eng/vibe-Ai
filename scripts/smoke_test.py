"""Checks that Python, PyTorch and NumPy work on this machine."""
import platform
import sys

import numpy as np
import torch


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(f"Need Python 3.10+, found {sys.version.split()[0]}")

    print(f"Python  : {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    print(f"PyTorch : {torch.__version__}")
    print(f"NumPy   : {np.__version__}")
    print(f"CPU threads used by torch: {torch.get_num_threads()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    torch.manual_seed(0)
    x = torch.randn(256, 256)
    y = x @ x.T
    assert y.shape == (256, 256)
    assert torch.isfinite(y).all()

    w = torch.randn(8, 1, requires_grad=True)
    loss = ((torch.randn(32, 8) @ w) ** 2).mean()
    loss.backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()

    print("Smoke test passed.")


if __name__ == "__main__":
    main()