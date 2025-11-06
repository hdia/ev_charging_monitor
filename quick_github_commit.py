#python quick_github_commit.py

#You will see:
#>> git add .
#>> git commit -m "Local update 2025-10-29 19:45:23"
#[main 7f3abc1] Local update 2025-10-29 19:45:23
#>> git push origin main
#✅ Done. Changes synced to GitHub.
# if you get a conflict message, use git pull --rebase origin main

import os
import subprocess
from datetime import datetime

# Optional: set your repo path here
REPO_DIR = r"C:\GitHub\ev_charging_monitor"


def run(cmd):
    """Run a shell command and print its output."""
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, cwd=REPO_DIR, check=False)

def main():
    os.chdir(REPO_DIR)

    # Create a timestamped commit message
    msg = f"Local update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Add all changes (tracked + new files)
    run("git add .")

    # Commit (ignore if nothing to commit)
    run(f'git commit -m "{msg}" || echo "No changes to commit"')

    # Push to main branch
    run("git push origin main")

    print("\n✅ Done. Changes synced to GitHub.\n")

if __name__ == "__main__":
    main()
