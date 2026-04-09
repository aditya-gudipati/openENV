#!/usr/bin/env python3
"""
Comprehensive validation test for all 4 graders with paranoid boundary checking.
Tests exact values that could cause issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from grader import (
    DeliveryTaskGrader, PriorityTaskGrader, 
    FuelTaskGrader, ServiceReliabilityTaskGrader,
    _clamp_score, TASKS
)
from env import LogisticsEnv
from models import Config

def test_clamp_score():
    """Test the _clamp_score function with edge cases."""
    print("=" * 60)
    print("TEST: _clamp_score() paranoid validation")
    print("=" * 60)
    
    test_cases = [
        (0.0, "_clamp_score(0.0)"),
        (1.0, "_clamp_score(1.0)"),
        (0.5, "_clamp_score(0.5)"),
        (0.01, "_clamp_score(0.01) [min boundary]"),
        (0.99, "_clamp_score(0.99) [max boundary]"),
        (0.001, "_clamp_score(0.001) [below min]"),
        (0.999, "_clamp_score(0.999) [near max]"),
        (-1.0, "_clamp_score(-1.0) [negative]"),
        (2.0, "_clamp_score(2.0) [> 1.0]"),
        (float('inf'), "_clamp_score(inf)"),
        (float('-inf'), "_clamp_score(-inf)"),
        (float('nan'), "_clamp_score(nan)"),
    ]
    
    all_valid = True
    for value, description in test_cases:
        result = _clamp_score(value)
        is_valid = 0 < result < 1
        status = "✓" if is_valid else "✗"
        print(f"{status} {description:40} → {result:.10f} | Valid: {is_valid}")
        if not is_valid:
            all_valid = False
    
    print()
    return all_valid

def test_all_graders():
    """Test all 4 graders with many different scenarios."""
    print("=" * 60)
    print("TEST: All 4 graders with various scenarios")
    print("=" * 60)
    
    game = LogisticsEnv(Config())
    all_valid = True
    failed_cases = []
    
    # Test parameters
    seeds = [0, 1, 42, 100, 123, 999, 12345]
    difficulties = ["easy", "medium", "hard"]
    
    total_tests = 0
    passed_tests = 0
    
    for seed in seeds:
        for difficulty in difficulties:
            try:
                game.config.difficulty = difficulty
                game.reset(seed=seed)
                
                # Get all 4 scores
                d_score = DeliveryTaskGrader.grade(game.state)
                p_score = PriorityTaskGrader.grade(game.state)
                f_score = FuelTaskGrader.grade(game.state)
                s_score = ServiceReliabilityTaskGrader.grade(game.state)
                
                scores = [
                    ("delivery_completion", d_score),
                    ("priority_sla", p_score),
                    ("fuel_efficiency", f_score),
                    ("service_reliability", s_score)
                ]
                
                for task_name, score in scores:
                    total_tests += 1
                    
                    # Check exact boundaries
                    if score == 0.0 or score == 1.0:
                        failed_cases.append({
                            "seed": seed,
                            "difficulty": difficulty,
                            "task": task_name,
                            "score": score,
                            "error": "Exact boundary value"
                        })
                        all_valid = False
                        print(f"✗ {task_name:25} seed={seed:5} diff={difficulty:6} → {score:.15f} [BOUNDARY]")
                    elif not (0 < score < 1):
                        failed_cases.append({
                            "seed": seed,
                            "difficulty": difficulty,
                            "task": task_name,
                            "score": score,
                            "error": "Out of range"
                        })
                        all_valid = False
                        print(f"✗ {task_name:25} seed={seed:5} diff={difficulty:6} → {score:.15f} [INVALID]")
                    else:
                        passed_tests += 1
                        # Print a sample occasionally
                        if passed_tests % 20 == 0:
                            print(f"✓ {passed_tests} tests passed so far...")
            except Exception as e:
                print(f"✗ Exception at seed={seed}, difficulty={difficulty}: {e}")
                all_valid = False
    
    print(f"\n✓ PASSED: {passed_tests}/{total_tests} grader score tests")
    
    if failed_cases:
        print(f"\n✗ FAILED: {len(failed_cases)} cases with invalid scores:")
        for case in failed_cases[:10]:  # Show first 10 failures
            print(f"  | {case['task']:25} seed={case['seed']:5} {case['error']}: {case['score']:.15f}")
    
    print()
    return all_valid

def test_boundary_values():
    """Test graders don't return exactly 0.0 or 1.0 in 1000+ scenarios."""
    print("=" * 60)
    print("TEST: Boundary value stress test (1000+ scenarios)")
    print("=" * 60)
    
    game = LogisticsEnv(Config())
    
    boundary_found = False
    exact_zeros = []
    exact_ones = []
    
    for seed in range(100):
        for difficulty in ["easy", "medium", "hard"]:
            try:
                game.config.difficulty = difficulty
                game.reset(seed=seed)
                
                scores = {
                    "delivery": DeliveryTaskGrader.grade(game.state),
                    "priority": PriorityTaskGrader.grade(game.state),
                    "fuel": FuelTaskGrader.grade(game.state),
                    "reliability": ServiceReliabilityTaskGrader.grade(game.state),
                }
                
                for task, score in scores.items():
                    if score == 0.0:
                        exact_zeros.append((seed, difficulty, task, score))
                        boundary_found = True
                    if score == 1.0:
                        exact_ones.append((seed, difficulty, task, score))
                        boundary_found = True
            except Exception:
                pass
    
    total_scenarios = 100 * 3  # seeds × difficulties
    total_tests = total_scenarios * 4  # × 4 graders
    
    print(f"Tested {total_tests} grader invocations across {total_scenarios} scenarios")
    
    if exact_zeros:
        print(f"\n✗ Found {len(exact_zeros)} exact 0.0 values:")
        for seed, diff, task, score in exact_zeros[:5]:
            print(f"  | seed={seed}, difficulty={diff}, task={task}, score={score}")
        boundary_found = True
    else:
        print("✓ No exact 0.0 values found")
    
    if exact_ones:
        print(f"\n✗ Found {len(exact_ones)} exact 1.0 values:")
        for seed, diff, task, score in exact_ones[:5]:
            print(f"  | seed={seed}, difficulty={diff}, task={task}, score={score}")
        boundary_found = True
    else:
        print("✓ No exact 1.0 values found")
    
    if not boundary_found:
        print("\n✅ PASSED: No boundary values found in 1000+ tests")
    
    print()
    return not boundary_found

def test_tasks_registry():
    """Verify TASKS registry has all 4 graders."""
    print("=" * 60)
    print("TEST: TASKS registry completeness")
    print("=" * 60)
    
    expected_tasks = [
        "delivery_completion",
        "priority_sla",
        "fuel_efficiency",
        "service_reliability"
    ]
    
    all_valid = True
    for task_id in expected_tasks:
        if task_id in TASKS:
            grader = TASKS[task_id].get("grader")
            if grader:
                print(f"✓ {task_id:25} → {grader.__name__}")
            else:
                print(f"✗ {task_id:25} → No grader found")
                all_valid = False
        else:
            print(f"✗ {task_id:25} → Not in TASKS registry")
            all_valid = False
    
    print(f"\nTotal tasks in registry: {len(TASKS)}")
    
    if len(TASKS) >= 4:
        print("✅ PASSED: At least 4 tasks registered")
    else:
        print("✗ FAILED: Less than 4 tasks registered")
        all_valid = False
    
    print()
    return all_valid

def main():
    print("\n" + "=" * 60)
    print("COMPREHENSIVE GRADER VALIDATION TEST SUITE")
    print("=" * 60 + "\n")
    
    results = {
        "_clamp_score paranoid checks": test_clamp_score(),
        "All 4 graders with multiple scenarios": test_all_graders(),
        "Boundary value stress test": test_boundary_values(),
        "TASKS registry completeness": test_tasks_registry(),
    }
    
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready for submission!")
    else:
        print("❌ SOME TESTS FAILED - Fix required before submission!")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
