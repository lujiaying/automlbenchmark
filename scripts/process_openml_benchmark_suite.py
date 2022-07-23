import numpy as np
import openml
import yaml

if __name__ == '__main__':
    # Load "AutoML Benchmark All Classification"
    benchmark_id = 271    # https://www.openml.org/search?type=benchmark&id=271
    benchmark = openml.study.get_suite(benchmark_id)
    
    tasks_df = openml.tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
    binary_tasks_df = tasks_df[tasks_df.NumberOfClasses <= 2]
    multiclass_tasks_df = tasks_df[tasks_df.NumberOfClasses > 2] 
    print(f'openml/s/{benchmark_id} #task={len(tasks_df)}: #binary={len(binary_tasks_df)}, #multiclass={len(multiclass_tasks_df)}')

    # split task collections into multiple benchmark.yaml
    output_file_dir = './resources/benchmarks'
    for idx, chunk_df in enumerate(np.array_split(binary_tasks_df, 4)):
        chunk_dict = chunk_df[['name', 'tid']].rename(columns={'tid':'openml_task_id'}).to_dict('records')
        output_file_name = f'openml_bench_{benchmark_id}-binary-chunk_{idx}.yaml'
        output_file_path = f'{output_file_dir}/{output_file_name}'
        with open(output_file_path, 'w') as fwrite:
            yaml.dump(chunk_dict, fwrite)
    for idx, chunk_df in enumerate(np.array_split(multiclass_tasks_df, 3)):
        chunk_dict = chunk_df[['name', 'tid']].rename(columns={'tid':'openml_task_id'}).to_dict('records')
        output_file_name = f'openml_bench_{benchmark_id}-multiclass-chunk_{idx}.yaml'
        output_file_path = f'{output_file_dir}/{output_file_name}'
        with open(output_file_path, 'w') as fwrite:
            yaml.dump(chunk_dict, fwrite)