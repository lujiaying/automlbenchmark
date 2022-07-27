import pandas as pd

def show_failed_tasks(df: pd.DataFrame):
    df = df[df.logloss.isnull()][['id', 'task', 'fold', 'info']]
    print(df)


def generate_mean_quality_speed(df: pd.DataFrame):
    df = df[df.logloss.notna()]
    print(df)
    df.logloss.mean()

if __name__ == '__main__':
    multiclass_ag_il1ms_path = '/home/ec2-user/automlbenchmark/results/autogluon_v0_5_1_high_il0_001.openml_bench_271-multiclass.1h8c.aws.20220724T190434/scores/results.csv'
     
    df = pd.read_csv(multiclass_ag_il1ms_path)
    # show_failed_tasks(df)
    generate_mean_quality_speed(df)