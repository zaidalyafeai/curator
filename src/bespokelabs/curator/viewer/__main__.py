import os
import subprocess
import sys
from pathlib import Path

def get_viewer_path():
    return str(Path(__file__).parent)

def ensure_dependencies():
    """Ensure npm dependencies are installed"""
    static_dir = os.path.join(get_viewer_path(), 'static')
    node_modules = os.path.join(static_dir, 'node_modules')
    
    if not os.path.exists(node_modules):
        print("First run: Installing Node.js dependencies...")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd=static_dir,
                check=True
            )
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("Error: Node.js and npm are required. Please install them first.")
            sys.exit(1)

def start_nextjs_server():
    viewer_path = get_viewer_path()
    static_dir = os.path.join(viewer_path, 'static')
    server_file = os.path.join(viewer_path, 'server.js')
    
    if not os.path.exists(os.path.join(static_dir, '.next')):
        print("Error: Next.js build artifacts not found. The package may not be built correctly.")
        sys.exit(1)
    
    try:
        env = os.environ.copy()
        env["NODE_ENV"] = "production"
        
        subprocess.run(
            ["node", server_file],
            cwd=viewer_path,
            env=env,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error starting Next.js server: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Node.js is not installed. Please install Node.js to run the viewer.")
        sys.exit(1)

def main():
    ensure_dependencies()
    start_nextjs_server()

if __name__ == "__main__":
    main()
