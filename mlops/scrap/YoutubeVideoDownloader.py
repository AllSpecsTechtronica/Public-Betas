#!/usr/bin/env python3
"""
Simple interactive YouTube video downloader using yt-dlp.
Enter a YouTube URL when prompted, choose a resolution,
and the video will be downloaded to the 'downloaded_videos' directory.
"""

import os
import subprocess
import sys
import json
from typing import Dict, List, Optional

def check_yt_dlp_installed() -> bool:
    """Check if yt-dlp is installed and available."""
    try:
        result = subprocess.run(['yt-dlp', '--version'], 
                              capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_yt_dlp():
    """Install yt-dlp using pip."""
    try:
        print("Installing yt-dlp...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp'], 
                      check=True)
        print("yt-dlp installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install yt-dlp: {e}")
        return False

def get_video_info(url: str) -> Optional[Dict]:
    """Get video information using yt-dlp."""
    try:
        print("[→] Fetching video information...")
        result = subprocess.run([
            'yt-dlp', '--dump-json', '--no-playlist', '--no-warnings', url
        ], capture_output=True, text=True, check=True, timeout=30)
        
        video_info = json.loads(result.stdout)
        return video_info
    except subprocess.TimeoutExpired:
        print("[!] Timeout while fetching video information. Please try again.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"[!] Failed to get video info: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"[!] Failed to parse video info: {e}")
        return None

def get_available_formats(video_info: Dict) -> List[Dict]:
    """Extract available formats from video info."""
    formats = []
    
    for fmt in video_info.get('formats', []):
        # Only include video formats with both video and audio
        if fmt.get('vcodec') != 'none' and fmt.get('acodec') != 'none':
            format_info = {
                'format_id': fmt.get('format_id', ''),
                'ext': fmt.get('ext', ''),
                'resolution': fmt.get('resolution', 'N/A'),
                'filesize': fmt.get('filesize', 0),
                'format_note': fmt.get('format_note', ''),
                'height': fmt.get('height', 0)
            }
            formats.append(format_info)
    
    # Sort by height (resolution) in descending order
    formats.sort(key=lambda x: x['height'], reverse=True)
    return formats

def download_video(url: str, output_path: str):
    """
    Downloads a YouTube video using yt-dlp.
    """
    print(f"\n[→] Processing URL: {url}")
    
    # Get video information
    video_info = get_video_info(url)
    if not video_info:
        print(f"[!] Failed to process URL '{url}'")
        return
    
    title = video_info.get('title', 'Unknown Title')
    print(f"\n[→] Video: '{title}'")
    
    # Get available formats
    formats = get_available_formats(video_info)
    
    if not formats:
        print("[!] No suitable video formats found.")
        return
    
    # Display available formats
    print("\nAvailable formats:")
    for i, fmt in enumerate(formats[:10], 1):  # Show top 10 formats
        filesize_mb = f"{fmt['filesize'] / (1024 * 1024):.2f} MB" if fmt['filesize'] else "N/A"
        print(f"  {i}: {fmt['resolution']} ({fmt['ext']}) - {filesize_mb}")
    
    # Let user choose format or use best quality
    choice = input(f"\nEnter choice (1-{min(10, len(formats))}), or press Enter for best quality: ").strip()
    
    selected_format = None
    if not choice:
        # Use best quality (first in the sorted list)
        selected_format = formats[0]
        print(f"[→] Using best quality: {selected_format['resolution']}")
    else:
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(formats):
                selected_format = formats[choice_idx]
            else:
                print("[!] Invalid choice. Using best quality.")
                selected_format = formats[0]
        except ValueError:
            print("[!] Invalid input. Using best quality.")
            selected_format = formats[0]
    
    # Download the video
    print(f"\n[→] Downloading: '{title}' @ {selected_format['resolution']}")
    
    try:
        # Use yt-dlp to download with the selected format
        cmd = [
            'yt-dlp',
            '-f', selected_format['format_id'],
            '-o', os.path.join(output_path, '%(title)s.%(ext)s'),
            '--no-playlist',
            url
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[✓] Download complete! Saved in '{os.path.abspath(output_path)}'")
        
    except subprocess.CalledProcessError as e:
        print(f"[!] Download failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")

def test_yt_dlp():
    """Test yt-dlp with a simple YouTube URL."""
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - short video
    print("Testing yt-dlp with a short YouTube video...")
    
    video_info = get_video_info(test_url)
    if video_info:
        title = video_info.get('title', 'Unknown')
        print(f"[✓] Successfully fetched video: '{title}'")
        return True
    else:
        print("[!] Failed to fetch video information")
        return False

def main():
    """
    Runs an interactive loop to download multiple YouTube videos.
    """
    # Check if yt-dlp is installed
    if not check_yt_dlp_installed():
        print("yt-dlp is not installed. Attempting to install...")
        if not install_yt_dlp():
            print("[!] Please install yt-dlp manually: pip install yt-dlp")
            return
    
    # Test yt-dlp functionality
    if not test_yt_dlp():
        print("[!] yt-dlp test failed. Please check your installation.")
        return
    
    output_dir = "downloaded_videos"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"[!] Failed to create download directory '{output_dir}': {e}")
            return
    
    print("--- YouTube Video Downloader ---")
    print(f"Videos will be saved to: {os.path.abspath(output_dir)}")
    print("Enter a YouTube video URL, choose a resolution, and start downloading.")
    print("Type 'q', 'quit', or 'exit' to stop.")

    while True:
        try:
            url = input("\n> YouTube URL: ")
            url = url.strip()
            if url.lower() in ['q', 'quit', 'exit']:
                print("\nExiting downloader. Goodbye!")
                break
            if not url:
                continue
            
            download_video(url, output_dir)
        except KeyboardInterrupt:
            print("\n\nExiting downloader. Goodbye!")
            break
        except Exception as e:
            print(f"[!] An unexpected error occurred in the main loop: {e}")

if __name__ == "__main__":
    main()
