#!/usr/bin/env python3
"""
downloadYolovWeights.py

Simple file downloader that accepts a URL and destination path.
Supports HTTP/HTTPS URLs (using requests/urllib) and mega.nz URLs (using mega-get).
"""

import argparse
import pathlib
import subprocess
import sys
from urllib.parse import urlparse


def download_file(url: str, destination: pathlib.Path) -> None:
    """Download a file from URL to the specified destination.
    
    Args:
        url: The URL to download from (HTTP/HTTPS or mega.nz)
        destination: Path where the file should be saved
    """
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if destination.exists():
        print(f"File already exists: {destination}")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("Download cancelled.")
            return
    
    parsed = urlparse(url)
    
    # Handle mega.nz URLs
    if 'mega.nz' in parsed.netloc:
        print(f"Downloading from mega.nz: {url}")
        print(f"Destination: {destination}")
        
        try:
            # Download to temp location in the destination's parent directory
            temp_dir = destination.parent
            before_files = set(temp_dir.glob('*'))
            
            # Run mega-get
            subprocess.run(['mega-get', url], cwd=str(temp_dir), check=True)
            
            # Find newly created file
            after_files = set(temp_dir.glob('*'))
            new_files = after_files - before_files
            
            if new_files:
                downloaded = max(new_files, key=lambda p: p.stat().st_mtime)
                if downloaded.resolve() != destination.resolve():
                    downloaded.rename(destination)
                print(f"✓ Downloaded successfully: {destination}")
            else:
                raise RuntimeError("No new file detected after mega-get")
                
        except FileNotFoundError:
            print("Error: 'mega-get' command not found. Please install megatools.")
            print("  sudo apt-get install megatools")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error: mega-get failed: {e}")
            sys.exit(1)
    
    # Handle HTTP/HTTPS URLs
    else:
        try:
            import urllib.request
            print(f"Downloading from: {url}")
            print(f"Destination: {destination}")
            
            urllib.request.urlretrieve(url, str(destination))
            print(f"✓ Downloaded successfully: {destination}")
            
        except Exception as e:
            print(f"Error downloading file: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download a file from a URL to a specified destination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://example.com/weights.pth ./weights/model.pth
  %(prog)s https://mega.nz/file/abc123 ./weights/yolov_weights.pth
        """)
    
    parser.add_argument('url', type=str, help='URL to download from')
    parser.add_argument('destination', type=str, help='Destination file path')
    
    args = parser.parse_args()
    
    download_file(args.url, args.destination)


if __name__ == "__main__":
    main()