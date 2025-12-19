"""
Test script for Bird Counting API
Run this after starting the main.py server
"""

import requests
import json
from pathlib import Path
import time
from dotenv import load_dotenv
import os

load_dotenv()
API_URL = os.getenv("API_URL")

if API_URL is None:
    raise ValueError("API_URL is not set in environment variables")

def test_health_check():
    """Test the health endpoint"""
    print("=" * 60)
    print("Testing /health endpoint...")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    return response.status_code == 200

def test_analyze_video(video_path: str):
    """Test the video analysis endpoint"""
    print("=" * 60)
    print("Testing /analyze_video endpoint...")
    print("=" * 60)
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return False
    
    print(f"Uploading video: {video_path}")
    print("Processing (this may take a few minutes)...\n")
    
    start_time = time.time()
    
    with open(video_path, 'rb') as video_file:
        files = {'file': video_file}
        data = {
            'conf_thresh': 0.25,
            'iou_thresh': 0.7
        }
        
        response = requests.post(
            f"{API_URL}/analyze_video",
            files=files,
            data=data
        )
    
    elapsed = time.time() - start_time
    
    print(f"Status Code: {response.status_code}")
    print(f"Processing Time: {elapsed:.2f} seconds\n")
    
    if response.status_code == 200:
        result = response.json()
        
        # Print summary
        print("Analysis Complete!")
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"\nVideo Info:")
        for key, value in result['video_info'].items():
            print(f"  • {key}: {value}")
        
        print(f"\nBird Counts:")
        counts = result['counts']
        if len(counts) > 0:
            print(f"  • Total frames analyzed: {len(counts)}")
            print(f"  • First frame count: {counts[0]['bird_count']}")
            print(f"  • Last frame count: {counts[-1]['bird_count']}")
            avg_count = sum(c['bird_count'] for c in counts) / len(counts)
            print(f"  • Average count: {avg_count:.1f}")
        
        print(f"\nWeight Estimates:")
        weights = result['weight_estimates']
        for key, value in weights.items():
            if key != 'calibration_note':
                print(f"  • {key}: {value}")
        
        print(f"\nTracked Birds (Sample):")
        for track_id, track_data in result['tracks_sample'].items():
            print(f"  • {track_id}: weight_index={track_data['avg_weight_index']}")
        
        print(f"\nOutput:")
        print(f"  • Annotated video: ./output/{result['artifacts']['annotated_video']}")
        
        # Save full JSON
        output_json = Path("test_result.json")
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  • Full JSON: {output_json}")
        
        print("\n" + "=" * 60)
        print("Calibration Note:")
        print("=" * 60)
        print(weights.get('calibration_note', 'See documentation'))
        
        return True
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def main():
    """Run all tests"""
    print("\nBird Counting API Test Suite\n")
    
    # Test 1: Health check
    health_ok = test_health_check()
    
    if not health_ok:
        print("Health check failed! Is the server running?")
        print("Start the server with: python main.py")
        return
    
    time.sleep(1)
    
    # Test 2: Video analysis
    print("\n" + "=" * 60)
    video_path = input("Enter path to test video (or press Enter to skip): ").strip()
    
    if video_path:
        test_analyze_video(video_path)
    else:
        print("\nTo test video analysis, use:")
        print('  python test_api.py')
        print("\n  Or use curl:")
        print('  curl -X POST "http://localhost:8000/analyze_video" \\')
        print('    -F "file=@your_video.mp4" \\')
        print('    -F "conf_thresh=0.25"')
    
    print("\nTesting complete!\n")

if __name__ == "__main__":
    main()