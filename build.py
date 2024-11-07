# build before releasing

import os
import subprocess
import shutil
import sys

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd, check=True)
    return result

def npm_install():
    print("Running npm install")
    run_command("npm install", cwd="bespoke-dataset-viewer")


def npm_build():
    print("Running npm build")
    run_command("npm run build", cwd="bespoke-dataset-viewer")
    print("Copying static files")
    source_dir = os.path.join("bespoke-dataset-viewer", "build")
    target_dir = os.path.join("src", "bespokelabs", "curator", "viewer", "static")
    shutil.rmtree(target_dir, ignore_errors=True)
    shutil.copytree(source_dir, target_dir)
    print(f"Copied static files from {source_dir} to {target_dir}")


def run_pytest():
    print("Running pytest")
    try:
        run_command("pytest", cwd="tests")
    except subprocess.CalledProcessError:
        print("Pytest failed. Aborting build.")
        sys.exit(1)


def main():
    npm_install()
    npm_build()
    run_pytest()
    print("Build completed successfully.")


if __name__ == "__main__":
    main()
