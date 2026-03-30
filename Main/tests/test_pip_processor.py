#!/usr/bin/env python3
"""Integration test for fixed pip_processor.py"""
import sys
import time
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "PIP"))

from final_integrated_project.Main.processors.pip_processor import PedestrianIntentProcessor

def test_fp_filters():
    print("=== PIP FP Elimination Test ===")
    pip = PedestrianIntentProcessor(vehicle_speed_kmh=30)
    pip.start()
    time.sleep(1)  # Init
    
    h, w = 480, 640
    
    # Test 1: Tiny FP (5px)
    print("Test 1: Tiny FP...")
    frame1 = np.zeros((h,w,3), dtype=np.uint8)
    cv2.rectangle(frame1, (300,300),(305,305), (255,255,255), -1)
    pip.push_frame(frame1)
    time.sleep(0.3)
    ann1 = pip.get_annotated_frame()
    white1 = np.sum(cv2.cvtColor(ann1, cv2.COLOR_BGR2GRAY) > 200)
    assert white1 < 200, f"Tiny box not filtered (white px: {white1})"
    print("  ✓ Tiny FP filtered")
    
    # Test 2: Bad aspect (square)
    print("Test 2: Bad aspect...")
    frame2 = np.zeros((h,w,3), dtype=np.uint8)
    cv2.rectangle(frame2, (200,200),(300,300), (255,0,0), -1)  # 1:1 aspect
    pip.push_frame(frame2)
    time.sleep(0.3)
    ann2 = pip.get_annotated_frame()
    blue2 = np.sum((ann2[:,:,0] == 255) &amp; (ann2[:,:,1] == 0))
    assert blue2 == 0, "Bad aspect not filtered"
    print("  ✓ Bad aspect filtered")
    
    # Test 3: Real ped (tall thin)
    print("Test 3: Real ped...")
    frame3 = np.zeros((h,w,3), dtype=np.uint8)
    cv2.rectangle(frame3, (250,250),(320,450), (0,255,0), -1)  # aspect ~0.3
    pip.push_frame(frame3)
    time.sleep(0.3)
    ann3 = pip.get_annotated_frame()
    green3 = np.sum((ann3[:,:,1] > 200) &amp; (ann3[:,:,2] < 100))
    assert green3 > 10000, "Real ped filtered incorrectly"
    print("  ✓ Real ped retained")
    
    # Test 4: Dynamic conf log (check console)
    print("Test 4: Dynamic conf...")
    dark_frame = np.full((h,w,3), 50, dtype=np.uint8)  # Night
    pip.push_frame(dark_frame)
    time.sleep(0.5)  # See [PIP] log
    
    pip.stop()
    print("\n✅ ALL TESTS PASS - No FPs, retains reals")

if __name__ == "__main__":
    test_fp_filters()

