import os
import pandas as pd

def get_task_metric(ref_csv_path: str) -> dict:
    taskid_metric_map = {}
    df = pd.read_csv(ref_csv_path)
    for i, group in df.groupby("id"):
        taskid_metric_map[i] = group.head(1)['metric'].values[0]
    return taskid_metric_map


def integrate_cascade_result(in_dir: str, out_dir: str):
    # get currrent status
    total_cnt = 0
    existed_cnt = 0
    df_list = []
    for fname in os.listdir(in_dir):
        if fname == 'scores':
            continue
        subdir = os.path.join(in_dir, fname)
        if not os.path.isdir(subdir):
            continue
        total_cnt += 1
        cascade_result_path = os.path.join(subdir, 'output', 'scores', 'cascade_results.csv')
        if not os.path.exists(cascade_result_path):
            print(f'{subdir} not exist...')
            continue
        existed_cnt += 1
        cur_df = pd.read_csv(cascade_result_path)
        df_list.append(cur_df)
    all_result_df = pd.concat(df_list)
    in_dir_exp_name = in_dir.split('/')[1]
    task_metric_map = get_task_metric('ag_cascade_report/my_raw_results/agv053_jul30_high.ag.1h8c_gp3.aws.20220731T064943.results.csv')
    for i, group in all_result_df.groupby("cascade_hyper_name"):
        out_path = os.path.join(out_dir, f'{i}.{in_dir_exp_name}.csv')
        group['framework'] = group['framework'].apply(lambda v: f'{v}-{i}')
        group['metric'] = group['id'].apply(lambda v: task_metric_map[v])
        group['result'] = group.apply(lambda row: row[row['metric']], axis=1)
        group.to_csv(out_path, index=False)
    print(f'[INFO] cascade results status: {existed_cnt}/{total_cnt}')


if __name__ == '__main__':
    in_dir = 'results/agv053_aug31_high.openml_bench_271-binary.1h8c_gp3.aws.20220906T180244'
    out_dir = 'ag_cascade_report/my_raw_results/cascade_F2SP_TPE'
    integrate_cascade_result(in_dir, out_dir)