# Copyright (c) EEEM071, University of Surrey

import argparse


def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument(
        "--root", type=str, default="./data", help="root path to data directory"
    )
        
    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        help="optimization algorithm (see optimizers.py)",
    )
    parser.add_argument(
        "--lr", default=0.03, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--weight-decay", default=5e-04, type=float, help="weight decay"
    )
    
    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument(
        "--max-epoch", default=20, type=int, help="maximum epochs to run"
    )
    
    parser.add_argument(
        "--train-batch-size", default=16, type=int, help="training batch size"
    )
    parser.add_argument(
        "--test-batch-size", default=16, type=int, help="test batch size"
    )
    return parser


def dataset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        "root": parsed_args.root,        
        "train_batch_size": parsed_args.train_batch_size,
        "test_batch_size": parsed_args.test_batch_size,      
        "epochs" : parsed_args.epochs, 
    }


def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in optimizers.py from
    the parsed command-line arguments.
    """
    return {
        "optim": parsed_args.optim,
        "lr": parsed_args.lr,
        "weight_decay": parsed_args.weight_decay,        
    }


