from curses.ascii import isdigit
import os
import shutil
import argparse
from typing import Tuple, List
from dataclasses import asdict

import pandas as pd
import openml
from autogluon.tabular import TabularPredictor, TabularDataset
import autogluon.core.metrics as metrics
from autogluon.tabular.predictor.cascade_do_no_harm import F2SP_Preset, GreedyP_Preset, CascadeConfig
from autogluon.tabular.predictor.cascade_do_no_harm import get_all_predecessor_model_names
from autogluon.core.utils.time import sample_df_for_time_func


def retrieve_train_test_data(res_path: str, ret_train: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    assert os.path.exists(res_path)
    df = pd.read_csv(res_path)
    assert len(df) == 1
    row = df.loc[0]
    task_id_full = row['id']
    task_id = int(task_id_full.split('/')[-1])
    fold = int(row['fold'])
    if task_id == 360975:
        # too big trigger out of memory in m5.2xlarge
        train_data = None
        test_data = pd.read_csv(f'/home/ec2-user/automlbenchmark/data/openml-360975_fold{fold}_test.csv')
        print(f'test_data load from disk file, shape={test_data.shape}')
    else:
        task = openml.tasks.get_task(task_id)
        print(task)
        all_data, _, _, _ = task.get_dataset().get_data()
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0)
        print(f'[DEBUG] fold={fold} train_data shape={train_indices.shape} test_data shape={test_indices.shape}')
        if ret_train:
            train_data = all_data.loc[train_indices]
        else:
            train_data = None
        test_data = all_data.loc[test_indices]
        print('[DEBUG] ret test_data in retrieve_train_test_data()')
    return train_data, test_data, row


def exec_fit_cascade(predictor: TabularPredictor, test_data: pd.DataFrame, 
                     cascade_algo_list: List[str],
                     infer_limit_batch_size: int, infer_limit_list: List[float],
                     ) -> pd.DataFrame:
    def get_cascade_config_WE_details(predictor: TabularPredictor, cascad_config: CascadeConfig):
        model_predecessors_dict = {}
        for model in cascad_config.model:
            model_predecessors = get_all_predecessor_model_names(predictor, model)
            model_predecessors_dict[model] = list(model_predecessors)
        return model_predecessors_dict

    # Start function
    metrics_mapping = dict(
        acc=metrics.accuracy,
        balacc=metrics.balanced_accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2,
        rmse=metrics.root_mean_squared_error,
    )
    metrics_mapping_r = {v.name: k for k, v in metrics_mapping.items()}
    cascade_results = []
    test_data_sampled = sample_df_for_time_func(df=test_data, sample_size=infer_limit_batch_size, 
                                                max_sample_size=infer_limit_batch_size)
    for infer_limit in infer_limit_list:
        for cascade_algo_name in cascade_algo_list:
            preset = F2SP_Preset() if cascade_algo_name == 'F2S+' else GreedyP_Preset()
            fit_cascade_params = {
                'raw_data_for_infer_speed': test_data,
                'infer_limit': infer_limit,
                'infer_limit_batch_size': infer_limit_batch_size,
                'hyperparameter_cascade': asdict(preset),
            }
            cascd_hyper_name = f'{cascade_algo_name}_{infer_limit}'    # used in result df to distinguish different trials
            cascade_config = predictor.fit_cascade(**fit_cascade_params)
            print(f'[DEBUG] {cascade_config}')
            if cascade_config is None:
                cascade_results.append(
                    {
                    'cascade_hyper_name': cascd_hyper_name,
                    }
                )
            else:
                infer_time, pred_probas = predictor.do_infer_with_cascade_conf(cascade_config, test_data)
                test_metrics = predictor.evaluate_predictions(test_data[predictor.label], pred_probas, silent=True)
                test_metrics = {metrics_mapping_r[k]: v for k, v in test_metrics.items() if k in metrics_mapping_r}
                infer_time_genuine, _ = predictor.do_infer_with_cascade_conf(cascade_config, test_data_sampled)
                print(f'[DEBUG] infer_time={infer_time}, genuine_time={infer_time_genuine}, test_metrics={test_metrics}')
                cascade_m_predecessors_dict = get_cascade_config_WE_details(predictor, cascade_config)
                cascade_results.append(
                    {
                    'cascade_hyper_name': cascd_hyper_name,
                    'training_duration': cascade_config.fit_time,
                    'predict_duration': infer_time,
                    'predict_duration_genuine': infer_time_genuine,
                    'sec_per_row': infer_time / len(test_data),
                    'genuine_sec_per_row': infer_time_genuine / infer_limit_batch_size,
                    **test_metrics,
                    'infer_limit': cascade_config.infer_limit,
                    'infer_limit_batch_size': cascade_config.infer_limit_batch_size,
                    'cascade_config': asdict(cascade_config),
                    'weighted_ensemble_info': cascade_m_predecessors_dict,
                    }
                )
    return pd.DataFrame.from_records(cascade_results)


def main(args: argparse.Namespace):
    # traversal dir
    for fname in os.listdir(args.amlb_result_dir):
        if fname == 'scores':
            continue
        subdir = os.path.join(args.amlb_result_dir, fname)
        if not os.path.isdir(subdir):
            continue
        subsubdir_model = os.path.join(subdir, 'output', 'models')
        print(f'[DEBUG] subsubdir_model={subsubdir_model}')
        assert os.path.exists(subsubdir_model)
        # task_name, fold = fname.split('.')[3:5]
        # assert isdigit(fold) == True
        temp = os.listdir(subsubdir_model)
        assert len(temp) == 1
        task_name = temp[0]
        temp = os.listdir(os.path.join(subsubdir_model, task_name))
        assert len(temp) == 1
        fold = temp[0]
        assert fold.isdigit() == True
        ckpt_dir = os.path.join(subsubdir_model, task_name, fold)
        model_archive_path = os.path.join(ckpt_dir, 'models.zip')
        assert os.path.exists(model_archive_path)
        if len(os.listdir(ckpt_dir)) == 1:
            shutil.unpack_archive(model_archive_path, ckpt_dir)
        # whether scores already exist
        cascade_result_path = os.path.join(subdir, 'output', 'scores', args.cascade_result_fname)
        if os.path.exists(cascade_result_path):
            continue
        # load data
        train_data, test_data, amlb_result_row = retrieve_train_test_data(os.path.join(subdir, 'output', 'results.csv'))
        metric_name = amlb_result_row['metric'] if amlb_result_row['metric'] != 'neg_logloss' else 'logloss'
        # load model and do fit_cascade
        predictor = TabularPredictor.load(ckpt_dir, require_version_match=False)
        cascade_results = exec_fit_cascade(predictor, test_data, args.cascade_algo_list, 
                args.infer_limit_batch_size, args.infer_limit_list)
        cascade_results['id'] = [amlb_result_row['id'] for _ in range(len(cascade_results))]
        cascade_results['task'] = [amlb_result_row['task'] for _ in range(len(cascade_results))]
        cascade_results['framework'] = [amlb_result_row['framework'] for _ in range(len(cascade_results))]
        cascade_results['constraint'] = [amlb_result_row['constraint'] for _ in range(len(cascade_results))]
        cascade_results['fold'] = [amlb_result_row['fold'] for _ in range(len(cascade_results))]
        cascade_results['type'] = [amlb_result_row['type'] for _ in range(len(cascade_results))]
        cascade_results['metric'] = [amlb_result_row['metric'] for _ in range(len(cascade_results))]
        cascade_results['score'] = [cascade_results[metric_name][_] for _ in range(len(cascade_results))]
        cascade_results.to_csv(cascade_result_path, index=False)
        print(f'[INFO] cascade results written into {cascade_result_path}')


def none_or_float(value: str):
    if value == "None":
        return None
    else:
        return float(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute fit_cascade() on saved artifacts")
    # required arguments
    parser.add_argument('--amlb_result_dir', type=str, required=True)
    # optional arguments
    parser.add_argument('--cascade_result_fname', type=str, default='cascade_results.csv')
    parser.add_argument('--cascade_algo_list', nargs="+", type=str, default=['F2S+'])
    parser.add_argument('--infer_limit_batch_size', type=int, default=10000)
    parser.add_argument('--infer_limit_list', nargs='+', type=none_or_float, default=[None, 5e-5, 3e-5, 1e-5, 5e-6, 3e-6])

    args = parser.parse_args()
    print(f'[INFO] Exp arguments {args}')
    main(args)