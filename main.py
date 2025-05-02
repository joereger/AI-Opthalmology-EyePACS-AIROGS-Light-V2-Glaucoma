#!/usr/bin/env python3
import sys
import os

print("Starting glaucoma detection pipeline...")

# Ensure the project root directory is in the Python path FIRST
# This allows imports like 'from src.cli import ...' when running main.py
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning
print("Python path configured.")

# Import the main CLI group from the cli module within the src package
print("Loading CLI module (this may take a moment)...")
try:
    from src.cli import main_cli
except ImportError as e:
    print(f"Error importing CLI application: {e}", file=sys.stderr)
    print("Please ensure 'src' directory is in the Python path or run from the project root.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    # Execute the CLI application
    # Click handles argument parsing and command execution
    main_cli()
