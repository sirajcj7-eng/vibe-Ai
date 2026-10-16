"""Commit and push chosen paths from a Kaggle notebook. Never prints the token.

    python scripts/kaggle_commit.py data/tokenizer results -m "message"
Needs the Kaggle secret GITHUB_TOKEN (fine-grained token, Contents: read and write).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("--branch", default="main")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is not set (add it as a Kaggle secret).")
        return 1
    url = run(["git", "remote", "get-url", "origin"]).stdout.strip()
    if not url.startswith("https://github.com/"):
        print("origin is not a https://github.com/ URL:", url)
        return 1
    push_url = url.replace("https://", f"https://x-access-token:{token}@", 1)

    run(["git", "config", "user.name", "Kaggle build"])
    run(["git", "config", "user.email", "kaggle-build@users.noreply.github.com"])
    for step in (["git", "add", *args.paths], ["git", "commit", "-m", args.message]):
        r = run(step)
        if r.returncode != 0:
            print((r.stdout + r.stderr).replace(token, "***"))
            return r.returncode
    r = run(["git", "pull", "--rebase", push_url, args.branch])
    if r.returncode != 0:
        print((r.stdout + r.stderr).replace(token, "***"))
        return r.returncode
    r = run(["git", "push", push_url, f"HEAD:{args.branch}"])
    print((r.stdout + r.stderr).replace(token, "***") or "pushed")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())