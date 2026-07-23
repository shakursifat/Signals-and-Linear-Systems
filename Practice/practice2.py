def analyze_sales_performance():
    # Input reading
    s = int(input())  # Number of salespersons
    m = int(input())  # Number of products
    
    # Read the sales matrix A
    a = []
    for _ in range(s):
        a.append(list(map(float, input().split())))
        
    # Read target sales vector
    target_sales = list(map(float, input().split()))
    
    # Read integer K
    k = int(input())
    
    # Task 1: Compute achievement percentage matrix
    p = []
    for i in range(s):
        row = []
        for j in range(m):
            percentage = 100.0 * (a[i][j] / target_sales[j])
            row.append(percentage)
        p.append(row)
        
    # Print Percentage Matrix
    print("Percentage Matrix")
    for row in p:
        print(" ".join(f"{val:.2f}" for val in row))
        
    # Task 2: Salesperson summary
    print("Salesperson Summary")
    salesperson_averages = []  # To track for top K
    
    for i in range(s):
        avg_pct = sum(p[i]) / m
        salesperson_averages.append((avg_pct, i))
        
        # Find best-performing product (highest percentage)
        best_prod = 0
        max_prod_pct = p[i][0]
        for j in range(1, m):
            if p[i][j] > max_prod_pct:
                max_prod_pct = p[i][j]
                best_prod = j
        print(f"Salesperson {i}: Average = {avg_pct:.2f} Best Product = {best_prod}")
        
    # Task 3: Product summary
    print("Product Summary")
    for j in range(m):
        col_sum = 0
        top_salesperson = 0
        max_salesperson_pct = p[0][j]
        
        for i in range(s):
            col_sum += p[i][j]
            if p[i][j] > max_salesperson_pct:
                max_salesperson_pct = p[i][j]
                top_salesperson = i
                
        prod_avg = col_sum / s
        print(f"Product {j}: Average = {prod_avg:.2f} Top Salesperson = {top_salesperson}")
        
    # Task 4: Print IDs of Top K salespersons based on average achievement
    print(f"Top {k} Salespersons")
    # Sort descending by average achievement percentage. 
    # If tied, index sorting defaults to insertion/original placement order.
    top_k = sorted(salesperson_averages, key=lambda x: x[0], reverse=True)[:k]
    for avg, idx in top_k:
        print(idx)
        
    # Task 5: Convert percentages to grades and count them
    grade_counts = {"Excellent": 0, "Good": 0, "Average": 0, "Poor": 0}
    for i in range(s):
        for j in range(m):
            val = p[i][j]
            if val >= 90:
                grade_counts["Excellent"] += 1
            elif val >= 75:
                grade_counts["Good"] += 1
            elif val >= 60:
                grade_counts["Average"] += 1
            else:
                grade_counts["Poor"] += 1
                
    print("Grade Count")
    print(f"Excellent: {grade_counts['Excellent']}")
    print(f"Good: {grade_counts['Good']}")
    print(f"Average: {grade_counts['Average']}")
    print(f"Poor: {grade_counts['Poor']}")

if __name__ == "__main__":
    analyze_sales_performance()