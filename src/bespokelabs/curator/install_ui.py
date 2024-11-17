"""Enhanced installation UI for bespokelabs-curator."""
import sys
import subprocess
from typing import Optional
from rich.console import Console
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel


def run_pip_install(package_spec: str) -> subprocess.CompletedProcess:
    """Run pip install and capture output."""
    process = subprocess.run(
        [sys.executable, "-m", "pip", "install", package_spec],
        capture_output=True,
        text=True,
        check=True
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
        "Installing ",
        (package_name, "cyan"),
        " ...\n",
        "↳ Your synthetic data journey begins in moments"
    )
    spinner = Spinner("dots", text=loading_text, style="green")
    
   # Create success text
    success_text = Text()
    success_text.append("✨ Curator installed successfully!\n\n", style="bold green")
    success_text.append("Start building production-ready synthetic data pipelines:\n\n", style="dim white")
    success_text.append("📚 Documentation: ", style="dim white")
    success_text.append("docs.bespokelabs.ai", style="dim cyan link https://docs.bespokelabs.ai")
    success_text.append("\n")
    success_text.append("📦 Repository: ", style="dim white")
    success_text.append("github.com/bespokelabsai/curator", style="dim cyan link https://github.com/bespokelabsai/curator")
    success_text.append("\n")
    success_text.append("💬 Get help: ", style="dim white")
    success_text.append("discord.gg/KqpXvpzVBS", style="dim cyan link https://discord.com/invite/KqpXvpzVBS")
    
    # Show live display that we'll update
    with Live(spinner, console=console, refresh_per_second=15) as live:
        try:
            # Run the installation
            process = run_pip_install(package_spec)
            # Update the display with success message
            live.update(success_text)
            
        except subprocess.CalledProcessError as e:
            # Create error text that will replace the spinner
            error_text = Text()
            error_text.append(e.stderr, style="red")
            # Update display with error message
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
