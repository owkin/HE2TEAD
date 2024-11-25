import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # MAIN ARGUMENTS
    main_args = parser.add_argument_group("args")
    main_args.add_argument(
        "--root", "-r", type=str, default="/shared/widy/data/MIL_benchmarks/TCGA/features_moco/", help="config name"
    )
    main_args.add_argument(
        "--config", "-c", type=str, default="def", help="config name"
    )
    main_args.add_argument(
        "--savedir", "-d", type=str, default="/workspace/sanofi_meso/experiments", help="directory to log runs"
    )

    # DATA ARGUMENTS
    data_args = parser.add_argument_group("data related")
    data_args.add_argument(
        "--cohorts",
        default="TCGA-MESO"
    )
    # MODEL ARGUMENTS
    model_args = parser.add_argument_group("model related")
    model_args.add_argument("--use_saved_models", default=None)
    model_args.add_argument(
        "--save_models", action="store_true", help="save trained models"
    )
    args = parser.parse_args()
    print(vars(args))
    return args

