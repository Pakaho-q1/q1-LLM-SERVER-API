import subprocess
import json
import os
from datetime import datetime
from pathlib import Path

CONFIG_FILE = "git-upload-config.json"


def run(cmd):
    return subprocess.run(cmd, shell=True)


def run_capture(cmd):
    return subprocess.check_output(cmd, shell=True).decode().strip()


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def ensure_git_repo():
    if not Path(".git").exists():
        print("Initializing git repository...")
        run("git init")


def detect_branch(config):
    try:
        branch = run_capture("git rev-parse --abbrev-ref HEAD")
        if branch == "HEAD":
            raise Exception
        return branch
    except:
        return config["default_branch"]


def create_gitignore(patterns):

    gitignore = Path(".gitignore")

    if not gitignore.exists():
        print("Creating .gitignore")

        with open(".gitignore", "w") as f:
            for p in patterns:
                f.write(p + "\n")


def apply_gitignore():

    print("Applying .gitignore rules...")

    run("git rm -r --cached .")
    run("git add .")


def detect_large_files(limit_mb):

    print("\nScanning for large files...")

    limit = limit_mb * 1024 * 1024

    for path in Path(".").rglob("*"):

        if path.is_file():

            size = path.stat().st_size

            if size > limit:
                print(f"Large file detected: {path} ({size/1024/1024:.2f}MB)")
                exit(1)


def commit_preview():

    print("\nCommit preview:\n")

    run("git status")
    print("")
    run("git diff --cached --stat")


def commit(config):

    if config["ask_for_message"]:

        msg = input("\nCommit message: ").strip()

        if not msg:
            msg = config["auto_commit_message"].format(
                date=datetime.now().strftime("%Y-%m-%d %H:%M")
            )

    else:

        msg = config["auto_commit_message"].format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )

    run(f'git commit -m "{msg}"')


def ensure_remote(remote):

    try:
        run_capture(f"git remote get-url {remote}")
    except:
        print("\nNo remote repository found")
        print("Add remote using:")
        print("git remote add origin git@github.com:USER/REPO.git")
        input("\nPress ENTER after adding remote...")


def push(remote, branch):

    print("\nPushing to remote...")

    run(f"git push -u {remote} {branch}")


def main():

    config = load_config()

    ensure_git_repo()

    branch = detect_branch(config)

    create_gitignore(config["ignore_patterns"])

    detect_large_files(config["large_file_limit_mb"])

    apply_gitignore()

    commit_preview()

    confirm = input("\nCommit? (y/n): ")

    if confirm.lower() != "y":
        print("Cancelled")
        return

    commit(config)

    ensure_remote(config["remote_name"])

    print("\nWaiting for SSH authentication if needed...")

    push(config["remote_name"], branch)

    print("\nDone.")


if __name__ == "__main__":
    main()
