import pandas as pd
import os

BASE_DIR = "MPCC_HF"
TASKS = {
    "Calendar Planning": "calendar_plan",
    "Meeting Planning": "meeting_plan",
    "Flight Planning": "flight_plan"
}
DIFFICULTIES = ["easy", "medium", "hard"]

def main():
    if not os.path.exists(BASE_DIR):
        print(f"Error: Directory '{BASE_DIR}' not found in current path.")
        return

    print(f"{'Task':<20} | {'Easy':<8} | {'Medium':<8} | {'Hard':<8} | {'Total':<8}")
    print("-" * 68)
    
    grand_total = 0
    
    for task_name, file_prefix in TASKS.items():
        counts = {}
        task_total = 0
        for diff in DIFFICULTIES:
            filename = f"{file_prefix}_{diff}.parquet"
            path = os.path.join(BASE_DIR, task_name, filename)
            
            count = 0
            if os.path.exists(path):
                try:
                    # Only read one column to make it faster if possible, but parquet reads are fast anyway
                    df = pd.read_parquet(path, columns=[]) 
                    count = len(df)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    # Fallback: try reading normally if columns=[] fails for some reason
                    try:
                        df = pd.read_parquet(path)
                        count = len(df)
                    except:
                        pass
            else:
                # Try checking if the file is named differently or path is slightly different?
                # Based on Glob result, paths seem correct: MPCC_HF/Calendar Planning/calendar_plan_easy.parquet
                print(f"Warning: File not found: {path}")
            
            counts[diff] = count
            task_total += count
        
        grand_total += task_total
        print(f"{task_name:<20} | {counts['easy']:<8} | {counts['medium']:<8} | {counts['hard']:<8} | {task_total:<8}")
    
    print("-" * 68)
    print(f"{'Grand Total':<20} | {'':<8} | {'':<8} | {'':<8} | {grand_total:<8}")

if __name__ == "__main__":
    main()
