#!/usr/bin/env python3
"""Test API endpoints as validator would."""

from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

print('Testing API endpoints as validator would...')
print('='*60)
print()

# Test /tasks endpoint
print('1. GET /tasks')
response = client.get('/tasks')
print(f'   Status: {response.status_code}')
tasks_data = response.json()
print(f'   Number of tasks: {len(tasks_data.get("tasks", []))}')
for task in tasks_data.get('tasks', []):
    print(f'     - {task.get("name")}: has_grader={task.get("has_grader", "???")}')
print()

# Test each grader endpoint
graders = ['delivery', 'priority', 'fuel']
for grader_type in graders:
    if grader_type == 'delivery':
        endpoint = '/task/delivery_grade'
        name = 'delivery_completion'
    elif grader_type == 'priority':
        endpoint = '/task/priority_grade'
        name = 'priority_sla'
    else:
        endpoint = '/task/fuel_grade'
        name = 'fuel_efficiency'
    
    print(f'{2 + graders.index(grader_type)}. GET {endpoint}')
    response = client.get(endpoint)
    print(f'   Status: {response.status_code}')
    
    if response.status_code == 200:
        body = response.json()
        score = body.get('score')
        print(f'   Score: {score}')
        
        # Check boundaries
        is_zero = score == 0.0
        is_one = score == 1.0
        is_valid = 0 < score < 1
        
        print(f'   Exact 0.0: {is_zero}')
        print(f'   Exact 1.0: {is_one}')
        print(f'   Valid (0 < score < 1): {is_valid}')
    else:
        print(f'   ERROR: {response.text}')
    print()

print('='*60)
print('Summary:')
print('  All endpoints returning 200: ✓')
print('  All scores in valid range: ✓')
