from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parent / 'runs' / 'detect'
teacher_csv = base_dir / 'teacher_model' / 'results.csv'
student_csv = base_dir / 'student_model' / 'results.csv'

def get_best_metrics(csv_path, label):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f'Missing results file: {csv_path}')
        return None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Find epoch with highest mAP50
    best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
    return {
        'Model': label,
        'Precision': round(best_row['metrics/precision(B)'], 4),
        'Recall': round(best_row['metrics/recall(B)'], 4),
        'mAP50': round(best_row['metrics/mAP50(B)'], 4),
        'mAP50-95': round(best_row['metrics/mAP50-95(B)'], 4)
    }

teacher_res = get_best_metrics(teacher_csv, 'Teacher (Baseline)')
student_res = get_best_metrics(student_csv, 'Student (Boosted)')

if teacher_res and student_res:
    summary = pd.DataFrame([teacher_res, student_res])
    print('\n================ MODEL PERFORMANCE COMPARISON ================')
    print(summary.to_string(index=False))
    print('==============================================================\n')
