# study_planner.py
# Simple AI-powered study planner: asks user for subjects and available hours,
# then generates a prioritized schedule using difficulty weights.
import json, math

DEFAULT_SUBJECTS = {
    'Math': 5,
    'Physics': 4,
    'Computer Science': 3,
    'English': 2,
    'History': 1
}

def generate_plan(subjects, hours):
    # Cycle through subjects sorted by difficulty so higher difficulty gets more slots
    sorted_subs = sorted(subjects.items(), key=lambda x: -x[1])
    plan = []
    for i in range(hours):
        plan.append(sorted_subs[i % len(sorted_subs)][0])
    return plan

def main():
    print('AI Study Planner')
    use_default = input('Use default subjects? (y/n): ').strip().lower()
    if use_default == 'y':
        subjects = DEFAULT_SUBJECTS
    else:
        subjects = {}
        n = int(input('How many subjects? '))
        for _ in range(n):
            name = input('Subject name: ').strip()
            diff = int(input('Difficulty (1-5): ').strip())
            subjects[name] = diff
    hours = int(input('How many hours available? '))
    plan = generate_plan(subjects, hours)
    schedule = [{'hour': i+1, 'subject': s} for i,s in enumerate(plan)]
    print('Generated Study Plan:')
    print(json.dumps(schedule, indent=2))

if __name__ == '__main__':
    main()
