def analyze_patient_records():
    # Read the number of patient visits
    n = int(input())
    
    # Task 1: Store every patient visit as a tuple inside a list
    visits = []
    for _ in range(n):
        record = input().split()
        patient_name = record[0]
        severity = int(record[1])
        treatment_time = int(record[2])
        visits.append((patient_name, severity, treatment_time))
        
    # Task 2: Build a dictionary for aggregations
    patient_summary = {}
    for name, severity, time in visits:
        if name not in patient_summary:
            patient_summary[name] = {'total_time': 0, 'total_severity': 0}
        patient_summary[name]['total_time'] += time
        patient_summary[name]['total_severity'] += severity
        
    # Task 3: Print the summary for every patient in alphabetical order
    sorted_patients = sorted(patient_summary.keys())
    for name in sorted_patients:
        total_time = patient_summary[name]['total_time']
        total_severity = patient_summary[name]['total_severity']
        print(f"{name} {total_time} {total_severity}")
        
    # Task 4: Print the patient having the highest total severity score.
    # Sorting alphabetically first inherently handles the lexicographical tie-breaker 
    # when we strictly use '>' to find the maximum.
    max_severity = -1
    top_patient = ""
    for name in sorted_patients:
        if patient_summary[name]['total_severity'] > max_severity:
            max_severity = patient_summary[name]['total_severity']
            top_patient = name
            
    print(f"TOP {top_patient} {max_severity}")

if __name__ == "__main__":
    analyze_patient_records()