#!/usr/bin/env python3
"""
Quick verification script to test that all graders work correctly.
Run this before submitting to verify Phase 2 validation will pass.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from env import LogisticsEnv
from models import Config
from grader import DeliveryTaskGrader, PriorityTaskGrader, FuelTaskGrader, TASKS

def test_graders():
    """Test all graders return valid scores in (0, 1)."""
    print("=" * 60)
    print("Testing All Graders for Meta PyTorch Hackathon")
    print("=" * 60)
    
    # Check TASKS registry
    print(f"\n✓ Found {len(TASKS)} tasks in TASKS registry:")
    for task_id, task_info in TASKS.items():
        print(f"  - {task_id}: {task_info['grader'].__name__}")
    
    if len(TASKS) < 3:
        print("\n❌ ERROR: Must have at least 3 tasks with graders!")
        return False
    
    # Initialize environment
    print("\n✓ Initializing environment...")
    config = Config(difficulty="medium", seed=42)
    env = LogisticsEnv(config)
    env.reset()
    
    # Test each grader
    print("\n✓ Testing grader outputs:")
    print("-" * 60)
    
    all_valid = True
    
    try:
        delivery_score = float(DeliveryTaskGrader.grade(env.state))
        print(f"  DeliveryTaskGrader:  {delivery_score:.4f}", end="")
        if not (0 < delivery_score < 1):
            print(" ❌ OUT OF RANGE!")
            all_valid = False
        else:
            print(" ✓")
    except Exception as e:
        print(f" ❌ ERROR: {e}")
        all_valid = False
    
    try:
        priority_score = float(PriorityTaskGrader.grade(env.state))
        print(f"  PriorityTaskGrader:  {priority_score:.4f}", end="")
        if not (0 < priority_score < 1):
            print(" ❌ OUT OF RANGE!")
            all_valid = False
        else:
            print(" ✓")
    except Exception as e:
        print(f" ❌ ERROR: {e}")
        all_valid = False
    
    try:
        fuel_score = float(FuelTaskGrader.grade(env.state))
        print(f"  FuelTaskGrader:      {fuel_score:.4f}", end="")
        if not (0 < fuel_score < 1):
            print(" ❌ OUT OF RANGE!")
            all_valid = False
        else:
            print(" ✓")
    except Exception as e:
        print(f" ❌ ERROR: {e}")
        all_valid = False
    
    print("-" * 60)
    
    # Summary
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ ALL TESTS PASSED!")
        print("\nYour submission should now pass Phase 2 validation:")
        print("  ✓ 3 tasks with graders found")
        print("  ✓ All scores strictly between 0 and 1")
        print("\nReady to resubmit! 🚀")
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease fix the issues before resubmitting.")
    print("=" * 60)
    
    return all_valid

if __name__ == "__main__":
    success = test_graders()
    sys.exit(0 if success else 1)
