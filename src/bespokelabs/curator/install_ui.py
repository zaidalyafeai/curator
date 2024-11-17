"""Enhanced installation UI for bespokelabs-curator."""
import sys
import subprocess
from typing import Optional
from rich.console import Console
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel
from rich.progress import ProgressBar

def create_progress_bar(completed: float = 0) -> Text:
    """Create a stylish progress bar with the given completion percentage."""
    width = 40  # Increased width for better visual
    filled = int(width * completed)
    bar = Text()
    bar.append("\n╭", style="dim white")
    bar.append("─" * (width + 2), style="dim white")
    bar.append("╮\n", style="dim white")
    bar.append("│ ", style="dim white")
    bar.append("█" * filled, style="green")
    bar.append("▒" * (width - filled), style="dim white")
    bar.append(" │", style="dim white")
    bar.append(f"\n╰", style="dim white")
    bar.append("─" * (width + 2), style="dim white")
    bar.append("╯", style="dim white")
    return bar

def run_pip_install(package_spec: str) -> subprocess.Popen:
    """Run pip install and capture output."""
    process = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", package_spec],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    return process

def enhanced_install(package_name: str, version: Optional[str] = None) -> None:
    """
    Enhance pip installation with a professional progress UI.
    
    Args:
        package_name: Name of the package to install
        version: Optional specific version to install
    """
    console = Console()
    package_spec = f"{package_name}=={version}" if version else package_name
    
    # Create a properly styled spinner with installation message
    loading_text = Text.assemble(
        ("✨ Installing ", "bold"),
        (package_name, "bold cyan"),
        "\n",
        ("↳ ", "dim white"),
        ("Your synthetic data journey begins in moments", "dim white"),
        create_progress_bar(0),
        ("\n ", ""),
        ("Preparing your environment...", "italic dim white")
    )
    spinner = Spinner("dots2", text=loading_text, style="green")  # Using dots2 for smoother animation
    
    # Create success text with a box
    success_text = Text()
    success_text.append("✨ Curator installed successfully!\n\n", style="bold green")
    success_text.append("Start building production-ready synthetic data pipelines:\n\n", style="dim white")
    success_text.append("   📚 ", style="")
    success_text.append("docs.bespokelabs.ai", style="dim cyan link https://docs.bespokelabs.ai")
    success_text.append("\n   📦 ", style="")
    success_text.append("github.com/bespokelabsai/curator", style="dim cyan link https://github.com/bespokelabsai/curator")
    success_text.append("\n   💬 ", style="")
    success_text.append("discord.gg/KqpXvpzVBS", style="dim cyan link https://discord.com/invite/KqpXvpzVBS")
    
    # Show live display that we'll update
    with Live(spinner, console=console, refresh_per_second=30) as live:
        try:
            # Run the installation
            process = run_pip_install(package_spec)
            
            # Track progress through pip output
            current_step = "Preparing..."
            progress = 0.0
            while True:
                output_line = process.stdout.readline()
                if output_line == '' and process.poll() is not None:
                    break
                    
                # Parse pip output to update progress
                line = output_line.strip().lower()
                if "collecting" in line:
                    current_step = "Downloading packages..."
                    progress = 0.2
                elif "downloading" in line:
                    current_step = "Downloading packages..."
                    # Extract download progress if available
                    if "%" in line:
                        try:
                            percent = float(line.split("%")[0].split()[-1])
                            # Scale download progress between 20% and 60%
                            progress = 0.2 + (percent / 100.0 * 0.4)
                        except:
                            pass
                elif "installing" in line:
                    current_step = "Installing..."
                    progress = 0.7
                elif "successfully installed" in line:
                    current_step = "Finalizing..."
                    progress = 0.9
                
                # Update loading text with current progress
                loading_text = Text.assemble(
                    ("✨ Installing ", "bold"),
                    (package_name, "bold cyan"),
                    "\n",
                    ("↳ ", "dim white"),
                    ("Your synthetic data journey begins in moments", "dim white"),
                    create_progress_bar(progress),
                    ("\n ", ""),
                    (current_step, "italic dim white")
                )
                spinner.text = loading_text
            
            # Show completion
            loading_text = Text.assemble(
                ("✨ Installing ", "bold"),
                (package_name, "bold cyan"),
                "\n",
                ("↳ ", "dim white"),
                ("Your synthetic data journey begins in moments", "dim white"),
                create_progress_bar(1.0),
                ("\n ", ""),
                ("Installation complete!", "bold green")
            )
            spinner.text = loading_text
            
            # Check if installation was successful
            if process.poll() == 0:
                live.update(success_text)
            else:
                error = process.stderr.read()
                error_text = Text()
                error_text.append(error, style="red")
                live.update(error_text)
                sys.exit(1)
                
        except Exception as e:
            # Create error text for general errors
            error_text = Text()
            error_text.append(f"Error: {str(e)}", style="red")
            # Update display with error message
            live.update(error_text)
            sys.exit(1)
    
    # Add final newline
    console.print()


if __name__ == "__main__":
    enhanced_install("bespokelabs-curator")
