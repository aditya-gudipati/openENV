#!/usr/bin/env python3
"""List all defined tasks."""

from grader import TASKS

print('Tasks defined in your code:')
print('='*60)
print(f'Total tasks: {len(TASKS)}')
print()

task_list = [
    ('delivery_completion', TASKS.get('delivery_completion')),
    ('priority_sla', TASKS.get('priority_sla')),
    ('fuel_efficiency', TASKS.get('fuel_efficiency'))
]

for idx, (task_id, task_info) in enumerate(task_list, 1):
    if task_info:
        grader = task_info.get('grader')
        grader_name = grader.__name__ if grader else 'NO_GRADER'
        print(f'{idx}. {task_id}')
        print(f'   Grader: {grader_name}')
        print(f'   Description: {task_info.get("description")[:60]}...')
        print()
    else:
        print(f'{idx}. {task_id} - NOT FOUND')
        print()

print('='*60)
if len(TASKS) >= 3:
    print(f'✅ You have {len(TASKS)} tasks with graders (requirement: at least 3)')
else:
    print(f'❌ You only have {len(TASKS)} tasks (need at least 3)')
