import logging
import os
import shutil
import warnings
import sys
import tempfile
warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
import pandas as pd
matplotlib.use('agg')  # no need for tk

from autogluon.tabular import TabularPredictor
from autogluon.core.utils.savers import save_pd, save_pkl
import autogluon.core.metrics as metrics
from autogluon.tabular.version import __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path

log = logging.getLogger(__name__)


def get_model_true_infer_speed_per_row_batch(
         data,
         *,
         predictor,
         batch_size: int = 100000,
         repeats=1,
         silent=False):
     """
     Source: https://github.com/awslabs/autogluon/pull/1929

     Get per-model true inference speed per row for a given batch size of data.
     Parameters
     ----------
     data : :class:`TabularDataset` or :class:`pd.DataFrame`
         Table of the data, which is similar to a pandas DataFrame.
         Must contain the label column to be compatible with leaderboard call.
     predictor : TabularPredictor
         Fitted predictor to get inference speeds for.
     batch_size : int, default = 100000
         Batch size to use when calculating speed. `data` will be modified to have this many rows.
         If simulating large-scale batch inference, values of 100000+ are recommended to get genuine throughput estimates.
     repeats : int, default = 1
         Repeats of calling leaderboard. Repeat times are averaged to get more stable inference speed estimates.
     silent : False
         If False, logs information regarding the speed of each model + feature preprocessing.
     Returns
     -------
     time_per_row_df : pd.DataFrame, time_per_row_transform : float
         time_per_row_df contains each model as index.
             'pred_time_test_with_transform' is the end-to-end prediction time per row in seconds if calling `predictor.predict(data, model=model)`
             'pred_time_test' is the end-to-end prediction time per row in seconds minus the global feature preprocessing time.
             'pred_time_test_marginal' is the prediction time needed to predict for this particular model minus dependent model inference times and global preprocessing time.
         time_per_row_transform is the time in seconds per row to do the feature preprocessing.
     """
     import copy
     import time
     import numpy as np
     import pandas as pd
     data_batch = copy.deepcopy(data)
     len_data = len(data_batch)
     if len_data == batch_size:
         pass
     elif len_data < batch_size:
         # add more rows
         duplicate_count = int(np.ceil(batch_size / len_data))
         data_batch = pd.concat([data_batch for _ in range(duplicate_count)])
         len_data = len(data_batch)
     if len_data > batch_size:
         # sample rows
         data_batch = data_batch.sample(n=batch_size, random_state=0)
         len_data = len(data_batch)

     if len_data != batch_size:
         raise AssertionError(f'len(data_batch) must equal batch_size! ({len_data} != {batch_size})')

     predictor.persist_models(models='all')

     ts = time.time()
     for i in range(repeats):
         predictor.transform_features(data_batch)
     time_transform = (time.time() - ts) / repeats

     leaderboards = []
     for i in range(repeats):
         leaderboard = predictor.leaderboard(data_batch, silent=True)
         leaderboard = leaderboard[leaderboard['can_infer']][['model', 'pred_time_test', 'pred_time_test_marginal']]
         leaderboard = leaderboard.set_index('model')
         leaderboards.append(leaderboard)
     leaderboard = pd.concat(leaderboards)
     time_per_batch_df = leaderboard.groupby(level=0).mean()
     time_per_batch_df['pred_time_test_with_transform'] = time_per_batch_df['pred_time_test'] + time_transform
     time_per_row_df = time_per_batch_df / batch_size
     time_per_row_transform = time_transform / batch_size

     if not silent:
         for index, row in time_per_row_df.iterrows():
             time_per_row = row['pred_time_test_with_transform']
             time_per_row_print = time_per_row
             unit = 's'
             if time_per_row_print < 1e-2:
                 time_per_row_print *= 1000
                 unit = 'ms'
                 if time_per_row_print < 1e-2:
                     time_per_row_print *= 1000
                     unit = 'μs'
             print(f"{round(time_per_row_print, 3)}{unit} per row | {index}")
         time_per_row_transform_print = time_per_row_transform
         unit = 's'
         if time_per_row_transform_print < 1e-2:
             time_per_row_transform_print *= 1000
             unit = 'ms'
             if time_per_row_transform_print < 1e-2:
                 time_per_row_transform_print *= 1000
                 unit = 'μs'
         print(f"{round(time_per_row_transform_print, 3)}{unit} per row | transform_features")

     return time_per_row_df, time_per_row_transform


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

    train, test = dataset.train.path, dataset.test.path
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
            train_data=train,
            time_limit=config.max_runtime_seconds,
            **training_params
        )

    del train

    if is_classification:
        with Timer() as predict:
            probabilities = predictor.predict_proba(test, as_multiclass=True)
        predictions = probabilities.idxmax(axis=1).to_numpy()
    else:
        with Timer() as predict:
            predictions = predictor.predict(test, as_pandas=False)
        probabilities = None
    # Jiaying: add genueine infer duration/speed based on infer_limit_batch_size
    predict_genuine_duration = None
    if 'infer_limit_batch_size' in training_params:
        from autogluon.tabular import TabularDataset
        infer_limit_batch_size = training_params['infer_limit_batch_size']
        repeats = 1    # TODO: change to 5
        test_data_set = TabularDataset(test)
        log.info(f'Execute infer_util.get_model_true_infer_speed_per_row_batch() for infer_limit_batch_size={infer_limit_batch_size}')
        time_per_row_df, _ = get_model_true_infer_speed_per_row_batch(data=test_data_set, predictor=predictor,
                                                                      batch_size=infer_limit_batch_size, repeats=repeats, silent=True)
        # log.info(time_per_row_df)
        best_model_time_per_row = time_per_row_df.loc[predictor.get_model_best()]
        predict_genuine_duration = best_model_time_per_row['pred_time_test_with_transform'] * len(test_data_set)

    prob_labels = probabilities.columns.values.astype(str).tolist() if probabilities is not None else None

    _leaderboard_extra_info = config.framework_params.get('_leaderboard_extra_info', False)  # whether to get extra model info (very verbose)
    _leaderboard_test = config.framework_params.get('_leaderboard_test', False)  # whether to compute test scores in leaderboard (expensive)
    leaderboard_kwargs = dict(silent=True, extra_info=_leaderboard_extra_info)
    # Disabled leaderboard test data input by default to avoid long running computation, remove 7200s timeout limitation to re-enable
    if _leaderboard_test:
        leaderboard_kwargs['data'] = test

    leaderboard = predictor.leaderboard(**leaderboard_kwargs)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        log.info(leaderboard)

    num_models_trained = len(leaderboard)
    if predictor._trainer.model_best is not None:
        num_models_ensemble = len(predictor._trainer.get_minimum_model_set(predictor._trainer.model_best))
    else:
        num_models_ensemble = 1

    save_artifacts(predictor, leaderboard, config)
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
                  predict_genuine_duration=predict_genuine_duration)


def save_artifacts(predictor, leaderboard, config):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        if 'leaderboard' in artifacts:
            leaderboard_dir = output_subdir("leaderboard", config)
            save_pd.save(path=os.path.join(leaderboard_dir, "leaderboard.csv"), df=leaderboard)

        if 'info' in artifacts:
            ag_info = predictor.info()
            info_dir = output_subdir("info", config)
            save_pkl.save(path=os.path.join(info_dir, "info.pkl"), object=ag_info)

        if 'models' in artifacts:
            shutil.rmtree(os.path.join(predictor.path, "utils"), ignore_errors=True)
            models_dir = output_subdir("models", config)
            zip_path(predictor.path, os.path.join(models_dir, "models.zip"))
    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
