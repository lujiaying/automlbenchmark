from email.policy import default
import logging
from operator import le
import os
import shutil
import warnings
import sys
import tempfile
import time
from typing import Tuple
from dataclasses import asdict
warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
import pandas as pd
matplotlib.use('agg')  # no need for tk

from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.core.utils.savers import save_pd, save_pkl
import autogluon.core.metrics as metrics
from autogluon.tabular.version import __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path

log = logging.getLogger(__name__)


def get_predict_genuine_duration(predictor: TabularPredictor, training_params: dict, 
                                 leaderboard: pd.DataFrame, test_data: TabularDataset) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Genuine duration is calculated by resampling dataset size into **infer_limit_batch_size**
    """
    try:
        from autogluon.core.utils.infer_utils import get_model_true_infer_speed_per_row_batch
        default_batch_size = 10000
        n_repeats = 3
        infer_limit_batch_size = training_params.get('infer_limit_batch_size', default_batch_size)
        log.info(f'Execute infer_util.get_model_true_infer_speed_per_row_batch() for infer_limit_batch_size={infer_limit_batch_size}')
        time_per_row_df, _ = get_model_true_infer_speed_per_row_batch(data=test_data, predictor=predictor,
                                                                      batch_size=infer_limit_batch_size, repeats=n_repeats, silent=True)
        # cal best model genuine predict duration (end to end + feature transform)
        best_model_time_per_row = time_per_row_df.loc[predictor.get_model_best()]
        predict_genuine_duration = best_model_time_per_row['pred_time_test_with_transform'] * len(test_data)
        leaderboard = leaderboard.copy().set_index('model')
        # add genuine pred_time columns into leaderboard
        leaderboard.loc[time_per_row_df.index, 'genuine_pred_time_test'] = time_per_row_df['pred_time_test'] * len(test_data)
        leaderboard.loc[time_per_row_df.index, 'genuine_pred_time_test_with_transform'] = time_per_row_df['pred_time_test_with_transform'] * len(test_data)
        leaderboard.loc[time_per_row_df.index, 'genuine_pred_time_test_marginal'] = time_per_row_df['pred_time_test_marginal'] * len(test_data)
        leaderboard = leaderboard.reset_index()
        return leaderboard, predict_genuine_duration
    except ImportError:
        # log.info('get_model_true_infer_speed_per_row_batch() not exist, return predict_genuine_duration as None')
        print('get_model_true_infer_speed_per_row_batch() not exist, return predict_genuine_duration as None')
        return leaderboard, None


def execute_cascade_algorithm(predictor: TabularPredictor, test_data: TabularDataset) -> pd.DataFrame:
    from autogluon.tabular.predictor.cascade_do_no_harm import F2SP_Preset, GreedyP_Preset, CascadeConfig
    from autogluon.tabular.predictor.cascade_do_no_harm import get_all_predecessor_model_names
    from autogluon.core.utils.time import sample_df_for_time_func

    def get_cascade_config_WE_details(predictor: TabularPredictor, cascad_config: CascadeConfig):
        model_predecessors_dict = {}
        for model in cascad_config.model:
            model_predecessors = get_all_predecessor_model_names(predictor, model)
            model_predecessors_dict[model] = list(model_predecessors)
        return model_predecessors_dict

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
    infer_limit_batch_size = 10000
    test_data_sampled = sample_df_for_time_func(df=test_data, sample_size=infer_limit_batch_size, 
                                                max_sample_size=infer_limit_batch_size)
    #for infer_limit in [None, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]:
    for infer_limit in [None]:
        for cascade_algo_name in ['F2S+', 'Greedy+']:
            preset = F2SP_Preset() if cascade_algo_name == 'F2S+' else GreedyP_Preset()
            fit_cascade_params = {
                'infer_limit': infer_limit,
                'infer_limit_batch_size': infer_limit_batch_size,
                'hyperparameter_cascade': {f'{cascade_algo_name}_{infer_limit}': asdict(preset)},
            }
            cascd_train_duration_ts = time.time()
            cascade_configs_dict = predictor.fit_cascade(**fit_cascade_params)
            cascd_train_duration_te = time.time()
            for cascd_hyper_name, cascade_config in cascade_configs_dict.items():
                if cascade_config is None:
                    cascade_results.append(
                        {
                        'cascade_hyper_name': cascd_hyper_name,
                        'training_duration': cascd_train_duration_te - cascd_train_duration_ts,
                        }
                    )
                else:
                    infer_time, pred_probas = predictor.do_infer_with_cascade_conf(cascade_config, test_data)
                    test_metrics = predictor.evaluate_predictions(test_data[predictor.label], pred_probas, silent=True)
                    test_metrics = {metrics_mapping_r[k]: v for k, v in test_metrics.items() if k in metrics_mapping_r}
                    infer_time_genuine, _ = predictor.do_infer_with_cascade_conf(cascade_config, test_data_sampled)
                    #print(f'{cascd_hyper_name}, {cascade_config}, {infer_time}, {test_metrics}')
                    cascade_m_predecessors_dict = get_cascade_config_WE_details(predictor, cascade_config)
                    cascade_results.append(
                        {
                        'cascade_hyper_name': cascd_hyper_name,
                        'fit_cascade_params': fit_cascade_params,
                        'cascade_config': {**asdict(cascade_config), **{'WE_predecessors_info': cascade_m_predecessors_dict}},
                        'training_duration': cascd_train_duration_te - cascd_train_duration_ts,
                        'predict_duration': infer_time,
                        'predict_duration_genuine': infer_time_genuine,
                        'sec_per_row': infer_time / len(test_data),
                        'genuine_sec_per_row': infer_time_genuine / infer_limit_batch_size,
                        **test_metrics,
                        }
                    )
    return pd.DataFrame.from_records(cascade_results)


def run(dataset, config):
    log.info(f"\n**** AutoGluon [v{__version__}] ****\n")

    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2,
        rmse=metrics.root_mean_squared_error,
    )

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    is_classification = config.type == 'classification'
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    train_path, test_path = dataset.train.path, dataset.test.path
    label = dataset.target.name
    problem_type = dataset.problem_type

    models_dir = tempfile.mkdtemp() + os.sep  # passed to AG

    with Timer() as training:
        predictor = TabularPredictor(
            label=label,
            eval_metric=perf_metric.name,
            path=models_dir,
            problem_type=problem_type,
        ).fit(
            train_data=train_path,
            time_limit=config.max_runtime_seconds,
            **training_params
        )

    test_data = TabularDataset(test_path)
    # Persist model in memory that is going to be predicting to get correct inference latency
    predictor.persist_models('best')

    if is_classification:
        with Timer() as predict:
            probabilities = predictor.predict_proba(test_data, as_multiclass=True)
        predictions = probabilities.idxmax(axis=1).to_numpy()
    else:
        with Timer() as predict:
            predictions = predictor.predict(test_data, as_pandas=False)
        probabilities = None

    prob_labels = probabilities.columns.values.astype(str).tolist() if probabilities is not None else None

    _leaderboard_extra_info = config.framework_params.get('_leaderboard_extra_info', False)  # whether to get extra model info (very verbose)
    _leaderboard_test = config.framework_params.get('_leaderboard_test', False)  # whether to compute test scores in leaderboard (expensive)
    leaderboard_kwargs = dict(silent=True, extra_info=_leaderboard_extra_info)
    # Disabled leaderboard test data input by default to avoid long running computation, remove 7200s timeout limitation to re-enable
    if _leaderboard_test:
        leaderboard_kwargs['data'] = test_data

    leaderboard = predictor.leaderboard(**leaderboard_kwargs)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        log.info(leaderboard)

    num_models_trained = len(leaderboard)
    if predictor._trainer.model_best is not None:
        num_models_ensemble = len(predictor._trainer.get_minimum_model_set(predictor._trainer.model_best))
    else:
        num_models_ensemble = 1

    # Jiaying: add genueine infer duration/speed based on infer_limit_batch_size
    leaderboard_genuine, predict_genuine_duration = get_predict_genuine_duration(predictor, training_params, leaderboard, test_data)
    leaderboard = leaderboard_genuine
    # Jiaying: add cascade algorithm post training
    cascade_results = execute_cascade_algorithm(predictor, test_data)
    # End add

    save_artifacts(predictor, leaderboard, config, cascade_results)
    shutil.rmtree(predictor.path, ignore_errors=True)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  probabilities=probabilities,
                  probabilities_labels=prob_labels,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  models_ensemble_count=num_models_ensemble,
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  predict_genuine_duration=predict_genuine_duration,
                  )


def save_artifacts(predictor, leaderboard, config, cascade_results: pd.DataFrame):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard', 'cascade_results'])
    try:
        if 'leaderboard' in artifacts:
            leaderboard_dir = output_subdir("leaderboard", config)
            save_pd.save(path=os.path.join(leaderboard_dir, "leaderboard.csv"), df=leaderboard)

        if 'info' in artifacts:
            ag_info = predictor.info()
            info_dir = output_subdir("info", config)
            save_pkl.save(path=os.path.join(info_dir, "info.pkl"), object=ag_info)

        if 'models' in artifacts:
            #shutil.rmtree(os.path.join(predictor.path, "utils"), ignore_errors=True)
            models_dir = output_subdir("models", config)
            zip_path(predictor.path, os.path.join(models_dir, "models.zip"))

        if 'cascade_results' in artifacts:
            cascade_dir = output_subdir('cascade', config)
            save_pd.save(path=os.path.join(cascade_dir, "cascade_results.csv"), df=cascade_results)
    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
