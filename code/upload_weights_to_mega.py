#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload YOLOX trained weights to MEGA cloud storage
"""

import os
import sys
import argparse
from pathlib import Path
from mega import Mega


def find_latest_checkpoint(output_dir):
    """Find the best checkpoint file in the output directory"""
    output_path = Path(output_dir)
    
    # Look for best_ckpt.pth first, then latest_ckpt.pth
    best_ckpt = output_path / "best_ckpt.pth"
    latest_ckpt = output_path / "latest_ckpt.pth"
    
    if best_ckpt.exists():
        return best_ckpt
    elif latest_ckpt.exists():
        return latest_ckpt
    else:
        # Search for any .pth files
        pth_files = list(output_path.glob("*.pth"))
        if pth_files:
            # Return the most recent one
            return max(pth_files, key=lambda p: p.stat().st_mtime)
    
    return None


def upload_to_mega(file_path, mega_email, mega_password, remote_folder=None):
    """
    Upload a file to MEGA cloud storage
    
    Args:
        file_path: Path to the file to upload
        mega_email: MEGA account email
        mega_password: MEGA account password
        remote_folder: Optional remote folder name in MEGA
    """
    print(f"Uploading {file_path} to MEGA...")
    print(f"File size: {file_path.stat().st_size / (1024**3):.2f} GB")
    
    # Initialize MEGA
    mega = Mega()
    
    try:
        # Login to MEGA
        print("Logging in to MEGA...")
        m = mega.login(mega_email, mega_password)
        print("✓ Logged in successfully")
        
        # Get or create remote folder
        if remote_folder:
            print(f"Looking for folder: {remote_folder}")
            folder = m.find(remote_folder)
            
            if folder:
                print(f"✓ Found existing folder: {remote_folder}")
                # folder is a list, get the first match
                if isinstance(folder, list) and len(folder) > 0:
                    folder_node = folder[0]
                else:
                    folder_node = folder
            else:
                print(f"Creating folder: {remote_folder}")
                folder_node = m.create_folder(remote_folder)
                print(f"✓ Created folder: {remote_folder}")
        else:
            # Upload to root
            folder_node = None
        
        # Upload the file
        print(f"Uploading {file_path.name}...")
        if folder_node:
            uploaded_file = m.upload(str(file_path), folder_node[0] if isinstance(folder_node, list) else folder_node)
        else:
            uploaded_file = m.upload(str(file_path))
        
        print(f"✓ Upload completed successfully!")
        
        # Get shareable link
        link = m.get_upload_link(uploaded_file)
        print(f"\n{'='*60}")
        print(f"Shareable link: {link}")
        print(f"{'='*60}\n")
        
        return link
        
    except Exception as e:
        print(f"✗ Error during upload: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Upload YOLOX trained weights to MEGA cloud storage"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="YOLOX_outputs/yoloxl-visdrone",
        help="Output directory containing the checkpoint (default: YOLOX_outputs/yoloxl-visdrone)"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Specific checkpoint file to upload (overrides --output-dir)"
    )
    parser.add_argument(
        "-e", "--email",
        type=str,
        required=True,
        help="MEGA account email"
    )
    parser.add_argument(
        "-p", "--password",
        type=str,
        required=True,
        help="MEGA account password"
    )
    parser.add_argument(
        "-r", "--remote-folder",
        type=str,
        default="YOLOX_Weights",
        help="Remote folder name in MEGA (default: YOLOX_Weights)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload all .pth files in the output directory"
    )
    
    args = parser.parse_args()
    
    # Change to YOLOX directory
    yolox_dir = Path(__file__).parent.parent / "YOLOV"
    if yolox_dir.exists():
        os.chdir(yolox_dir)
        print(f"Changed directory to: {yolox_dir}")
    
    # Determine which file(s) to upload
    if args.file:
        # Specific file provided
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"✗ Error: File not found: {file_path}")
            sys.exit(1)
        files_to_upload = [file_path]
    elif args.all:
        # Upload all .pth files
        output_path = Path(args.output_dir)
        if not output_path.exists():
            print(f"✗ Error: Output directory not found: {output_path}")
            sys.exit(1)
        files_to_upload = list(output_path.glob("*.pth"))
        if not files_to_upload:
            print(f"✗ Error: No .pth files found in {output_path}")
            sys.exit(1)
    else:
        # Find best checkpoint
        checkpoint = find_latest_checkpoint(args.output_dir)
        if not checkpoint:
            print(f"✗ Error: No checkpoint found in {args.output_dir}")
            sys.exit(1)
        files_to_upload = [checkpoint]
    
    print(f"\n{'='*60}")
    print(f"Files to upload: {len(files_to_upload)}")
    for f in files_to_upload:
        print(f"  - {f}")
    print(f"{'='*60}\n")
    
    # Upload each file
    uploaded_links = []
    for file_path in files_to_upload:
        link = upload_to_mega(
            file_path,
            args.email,
            args.password,
            args.remote_folder
        )
        if link:
            uploaded_links.append((file_path.name, link))
        print()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Upload Summary:")
    print(f"{'='*60}")
    if uploaded_links:
        for filename, link in uploaded_links:
            print(f"✓ {filename}")
            print(f"  Link: {link}")
    else:
        print("✗ No files uploaded successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
