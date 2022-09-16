import os

import pandas as pd
import tqdm


if __name__ == '__main__':
    ori_dir = '/home/ec2-user/Datasets/cpp_benchmark_2021_60datasets'
    out_dir = '/home/ec2-user/Datasets/cpp_benchmark_2021_60datasets_for_automlbenchmark'
    for session_name in tqdm.tqdm(os.listdir(ori_dir)):
        session_dir = os.path.join(ori_dir, session_name)
        for subdir_name in os.listdir(session_dir):
            temp = os.listdir(os.path.join(session_dir, subdir_name))
            assert len(temp) == 1
            parquet_path = os.path.join(session_dir, subdir_name, temp[0])
            #print(parquet_path)
            df = pd.read_parquet(parquet_path)
            out_subdir = os.path.join(out_dir, session_name)
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir)
            out_path = os.path.join(out_subdir, f'{subdir_name}.csv')
            #print(out_path)
            # df.to_csv(out_path, index=False)