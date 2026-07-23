class Employee:
    # Class Attribute
    company_name = "Tech Corp"  # Placeholder name, editable as needed
    
    def __init__(self, name, employee_id):
        self.name = name
        self.employee_id = employee_id
        self.salaries = []
        
    def add_salary(self, salary):
        self.salaries.append(salary)
        
    def average_salary(self):
        if not self.salaries:
            return 0.0
        return sum(self.salaries) / len(self.salaries)
        
    def highest_salary(self):
        if not self.salaries:
            return 0
        return max(self.salaries)
        
    def annual_income(self):
        return sum(self.salaries)
        
    def __str__(self):
        return (f"Employee Name: {self.name}\n"
                f"Employee ID: {self.employee_id}\n"
                f"Average Salary: {self.average_salary():.2f}\n"
                f"Highest Salary: {self.highest_salary()}\n"
                f"Annual Income: {self.annual_income()}")

# Demonstration to complete Tasks 1, 2, and 3
if __name__ == "__main__":
    # 1. Create 5 Employee objects
    employees = [
        Employee("Alice", "E101"),
        Employee("Bob", "E102"),
        Employee("Charlie", "E103"),
        Employee("David", "E104"),
        Employee("Eva", "E105")
    ]
    
    # 2. Add 12 months of mock salary records for each employee
    # Giving Alice specific values to emulate the sample output
    mock_salaries = [45000, 46000, 48000, 52000, 44000, 43000, 45500, 47000, 49000, 43000, 44006, 51500]
    for salary in mock_salaries:
        employees[0].add_salary(salary)
        
    # Populate others with flat values for demonstration
    for emp in employees[1:]:
        for _ in range(12):
            emp.add_salary(50000)
            
    # 3. Display information for the first employee matching sample output format
    print(employees[0])