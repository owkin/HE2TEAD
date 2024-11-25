from pathlib import Path
import yaml
from rich import print
from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.utils.data import Subset, ConcatDataset
from he2tead.model import Chowder
from he2tead.engine.opts import get_args
from he2tead.engine.training import fit
from he2tead.data import TCGADataset


def run_cv(
    exp_id, root, cohorts=['TCGA_MESO'], config='def', save_models=False,
    n_repeats=1, n_folds=5, savedir='/workspace/sanofi_meso/experiments'
):

    mlflow_main_run = mlflow.start_run(experiment_id=exp_id, run_name="average")
    run_id = mlflow_main_run.info.run_id

    # Save directory
    cohorts_name = '_'.join(sorted(cohorts))
    save_dir = f"{savedir}/logs/{cohorts_name}/{config}/{run_id}"
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    logger.add(save_dir / "exp_{time}.log", format="{name} {message}", level="INFO")

    params = yaml.load(open("./configs.yaml", "rb"), Loader=yaml.FullLoader)[
        config
    ]
    model_params = params['model_params']
    training_params = params['training_params']
    logger.info(f"Config : {model_params}")

    dataset_pancancer = []
    ids = []
    y = []
    filenames = []
    df_all = []
    cohort_id = []
    for cohort in cohorts:

        cancer = cohort.split('_')[1]
        df = pd.read_csv(f'../../assets/activity_signature/{cancer}.csv', index_col=0)
        df['TEAD_500'] -= df['TEAD_500'].mean()
        df['TEAD_500'] /= df['TEAD_500'].std()

        _root = Path(root) / f'{cohort}'
        dataset = TCGADataset(df, _root)
        dataset_pancancer.append(dataset)

        df_all.append(dataset.df)
        ids.append(dataset.ids)
        y.append(dataset.y)
        filenames.append(dataset.filenames)
        cohort_id.append(np.repeat(cohort, len(dataset)))

    dataset = ConcatDataset(dataset_pancancer)
    df = pd.concat(df_all)
    ids = np.concatenate(ids)
    y = np.concatenate(y)
    cohort_id = np.concatenate(cohort_id)
    df_pred = df.loc[ids].reset_index()
    df_pred['filename'] = np.concatenate(filenames)
    df_pred['cohort'] = cohort_id

    mlflow.log_params(model_params)
    mlflow.log_params({'run_id': run_id})

    corrs = []
    p_vals = []
    cohorts_ = []

    for k in range(n_repeats):

        kf = StratifiedGroupKFold(n_splits=n_folds, random_state=k, shuffle=True)
        fold = 0

        for train_index, test_index in kf.split(ids, y[:, 0] > 0, groups=ids):

            mlflow_fold_run = mlflow.start_run(nested=True, experiment_id=exp_id)
            mlflow.log_params({
                'parent_run_id': run_id,
                'run_id': mlflow_fold_run.info.run_id,
                'run_type': 'fold',
                'repetition': k,
                'fold': fold,
            })

            split_save_dir = save_dir / f"split_{k}_fold_{fold}"
            split_save_dir.mkdir(exist_ok=True, parents=True)

            train_set = Subset(dataset, train_index)
            test_set = Subset(dataset, test_index)

            model = Chowder(**model_params)

            model.device = 'cuda'
            model.to(model.device)

            preds, labels = fit(
                model,
                train_set,
                test_set=test_set,
                params=training_params
            )

            df_pred.loc[test_index, f'pred split {k} fold {fold}'] = preds.flatten()
            df_test_tmp = df_pred.loc[test_index]
            cohort_group = df_test_tmp.groupby(df_test_tmp.index).first()['cohort']
            df_test_tmp = df_test_tmp.groupby(df_test_tmp.index).mean()
            df_test_tmp['cohort'] = cohort_group
            if save_models:
                dic = model.state_dict()
                torch.save(dic, f'{split_save_dir}/model.pt')
            for cohort in df_test_tmp['cohort'].unique():
                cohorts_.append(cohort)
                R, p = pearsonr(
                    df_test_tmp.loc[df_test_tmp['cohort'] == cohort, 'TEAD_500'],
                    df_test_tmp.loc[df_test_tmp['cohort'] == cohort, f'pred split {k} fold {fold}']
                )
                corrs.append(R)
                p_vals.append(p)
                logger.info(f'Split {k} fold {fold} cohort {cohort}, R: {R:.4f}, p-value: {p:.4f}')
                mlflow.log_metrics({'R': R, 'p-value': p})
            mlflow.end_run()

            fold += 1
        logger.info('--------------')

    pd.DataFrame.from_dict(params, orient="index").to_csv(
        save_dir / "config_params.csv"
    )
    pd.DataFrame.from_dict(args, orient="index").to_csv(
        save_dir / "args.csv"
    )

    all_val_metrics = pd.DataFrame(
        {'correlation': corrs, 'p_value': p_vals, 'cohort': cohorts_})
    all_val_metrics.to_csv(
        save_dir / "all_val_metrics.csv", header=True
    )
    df_pred.to_csv(save_dir / "all_test_preds.csv")
    logger.info(f"Params, Metrics and val predictions saved in {save_dir}")

    mlflow.log_metrics(
        {
            'R': np.mean(corrs),
            'p-value': np.mean(p_vals)
        }
    )
    mlflow.end_run()
    return corrs, p_vals


if __name__ == '__main__':

    args = get_args()

    # Mlflow Experiment
    exp_name = f"{args.cohorts}_{args.config}"
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp:
        exp_id = exp.experiment_id
    else:
        print(f"Experiment with name {exp_name} not found. Creating it.")
        exp_id = mlflow.create_experiment(name=exp_name)

    logger.info(vars(args))

    args = {
        'savedir': args.savedir,
        'root': args.root,
        'config': args.config,
        'cohorts': args.cohorts.split(','),
        'savedir': args.savedir,
        'save_models': args.save_models
    }

    corrs, p_vals = run_cv(exp_id=exp_id, **args)
