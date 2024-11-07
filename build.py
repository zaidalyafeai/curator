import os
import subprocess
import shutil
import sys
from pathlib import Path

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd, check=True)
    return result

def npm_install():
    print("Running npm install")
    run_command("npm install", cwd="bespoke-dataset-viewer")

def nextjs_build():
    print("Running Next.js build")
    run_command("npm run build", cwd="bespoke-dataset-viewer")
    print("Copying build artifacts to static folder")
    
    # Source and target directories
    source_base = Path("bespoke-dataset-viewer")
    target_base = Path("src/bespokelabs/curator/viewer/static")
    
    # Ensure target directory exists
    if target_base.exists():
        shutil.rmtree(target_base)
    target_base.mkdir(parents=True, exist_ok=True)

    # Copy only the necessary files, excluding node_modules
    files_to_copy = [
        '.next',
        'app',
        'components',
        'lib',
        'public',
        'types',
        'package.json',
        'package-lock.json',
        'next.config.ts',
        'next-env.d.ts',
        'tsconfig.json',
        'postcss.config.mjs',
        'tailwind.config.ts',
        'components.json'
    ]
    
    for item in files_to_copy:
        source = source_base / item
        target = target_base / item
        
        if source.exists():
            if source.is_file():
                shutil.copy2(source, target)
                print(f"Copied file {source} to {target}")
            elif source.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
                print(f"Copied directory {source} to {target}")
        else:
            print(f"Warning: {source} not found")

def run_pytest():
    print("Running pytest")
    try:
        run_command("pytest", cwd="tests")
    except subprocess.CalledProcessError:
        print("Pytest failed. Aborting build.")
        sys.exit(1)

def main():
    npm_install()
    nextjs_build()
    run_pytest()
    print("Build completed successfully.")

if __name__ == "__main__":
    main()