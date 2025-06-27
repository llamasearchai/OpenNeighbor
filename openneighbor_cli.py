#!/usr/bin/env python3
"""
OpenNeighbor CLI Wrapper Script

This script provides a direct way to run the OpenNeighbor CLI
if the console script entry point is not working.

Usage:
    python openneighbor_cli.py --help
    python openneighbor_cli.py demo
    python openneighbor_cli.py init my-project
    python openneighbor_cli.py train --epochs 10
    python openneighbor_cli.py recommend --user-id 42
    python openneighbor_cli.py serve
    python openneighbor_cli.py version

Author: Nik Jois <nikjois@llamasearch.ai>
"""

if __name__ == '__main__':
    from openneighbor.ui.cli import cli
    cli() 