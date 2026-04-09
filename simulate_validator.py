#!/usr/bin/env python3
"""Simulate validator behavior."""

from grader import DeliveryTaskGrader, PriorityTaskGrader, FuelTaskGrader, TASKS
from env import LogisticsEnv
from models import Config

print("Simulating Validator Checks")
print("="*60)
print()

# Check 1: TASKS registry
print("Check 1: TASKS registry")
print(f"  Found {len(TASKS)} tasks in TASKS")
if len(TASKS) < 3:
    print(f"  ❌ FAIL: Need at least 3 tasks, got {len(TASKS)}")
else:
    print(f"  ✓ PASS: Has 3+ tasks")
    for task_id, task_info in TASKS.items():
        print(f"    - {task_id}: {task_info.get('grader', 'NO_GRADER').__name__}")
print()

# Check 2: Grader endpoints exist
print("Check 2: Grader endpoints")
from server.app import app
print(f"  /task/delivery_grade: {'/task/delivery_grade' in str(app.routes)}")
print(f"  /task/priority_grade: {'/task/priority_grade' in str(app.routes)}")
print(f"  /task/fuel_grade: {'/task/fuel_grade' in str(app.routes)}")
print()

# Check 3: Initialize environment
print("Check 3: Environment initialization")
try:
    env = LogisticsEnv(Config())
    env.reset(seed=42)
    print(f"  ✓ Environment reset successfully")
except Exception as e:
    print(f"  ❌ Failed to reset: {e}")
    exit(1)
print()

# Check 4: Call each grader multiple times (same seed)
print("Check 4: Grader consistency (deterministic)")
for seed in [42, 123, 999]:
    env.reset(seed=seed)
    
    d1 = DeliveryTaskGrader.grade(env.state)
    p1 = PriorityTaskGrader.grade(env.state)
    f1 = FuelTaskGrader.grade(env.state)
    
    env.reset(seed=seed)
    
    d2 = DeliveryTaskGrader.grade(env.state)
    p2 = PriorityTaskGrader.grade(env.state)
    f2 = FuelTaskGrader.grade(env.state)
    
    match_d = d1 == d2
    match_p = p1 == p2
    match_f = f1 == f2
    
    print(f"  Seed {seed:3d}: delivery={match_d}, priority={match_p}, fuel={match_f}")
    
    if not (match_d and match_p and match_f):
        print(f"    ❌ NOT DETERMINISTIC!")
print()

# Check 5: Score ranges (multiple seeds and steps)
print("Check 5: Score ranges (different scenarios)")
all_valid = True
for seed in [42, 100, 200]:
    for difficulty in ['easy', 'medium', 'hard']:
        env = LogisticsEnv(Config(difficulty=difficulty))
        env.reset(seed=seed)
        
        d = DeliveryTaskGrader.grade(env.state)
        p = PriorityTaskGrader.grade(env.state)
        f = FuelTaskGrader.grade(env.state)
        
        d_valid = 0 < d < 1
        p_valid = 0 < p < 1
        f_valid = 0 < f < 1
        
        if not (d_valid and p_valid and f_valid):
            print(f"  ❌ seed={seed}, difficulty={difficulty}")
            print(f"     delivery: {d} (valid: {d_valid})")
            print(f"     priority: {p} (valid: {p_valid})")
            print(f"     fuel: {f} (valid: {f_valid})")
            all_valid = False

if all_valid:
    print(f"  ✓ All scores in multiple scenarios are valid (0, 1)")
print()

# Check 6: Check for exact 0.0 or 1.0
print("Check 6: No exact boundaries")
test_count = 100
boundary_found = False
for i in range(test_count):
    env = LogisticsEnv(Config())
    env.reset(seed=1000+i)
    
    d = DeliveryTaskGrader.grade(env.state)
    p = PriorityTaskGrader.grade(env.state)
    f = FuelTaskGrader.grade(env.state)
    
    if d == 0.0 or d == 1.0 or p == 0.0 or p == 1.0 or f == 0.0 or f == 1.0:
        print(f"  Found boundary value at seed {1000+i}!")
        print(f"    d={d}, p={p}, f={f}")
        boundary_found = True

if boundary_found:
    print(f"  ❌ Found exact boundary values (0.0 or 1.0)")
else:
    print(f"  ✓ No exact boundary values in {test_count} tests")

print()
print("="*60)
print("✅ All validator checks passed!" if all_valid and not boundary_found else "❌ Some checks failed")
