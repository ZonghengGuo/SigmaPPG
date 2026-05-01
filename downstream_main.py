import argparse
import sys
import os

from downstream.bp.preprocess import PreprocessBP
from downstream.bp.train import BPTrain
from downstream.vtac.preprocess import PreprocessVtac
from downstream.vtac.train import VtacTrainer
from downstream.stanford.preprocess import PreprocessStanford
from downstream.stanford.train import StanfordTrainer
from downstream.dalia.preprocess import PreprocessDalia
from downstream.dalia.train import DaliaTrainer
from downstream.bidmc.preprocess import PreprocessBIDMC
from downstream.bidmc.train import BIDMCTrainer
from downstream.butppg.preprocess import PreprocessButPPG
from downstream.butppg.train import ButPPGTrainer
from downstream.wesad.wesad_preprocess import PreprocessWESAD
from downstream.wesad.wesad_train import WESADTrainer
from downstream.ppgbp.ppgbp_preprocessing import PreprocessPPGBP
from downstream.ppgbp.ppgbp_train import PPGBPTrainer
from downstream.sdb.sdb_preprocessing import PreprocessSDB
from downstream.sdb.sdb_train import SDBTrainer
from downstream.real_world.rw_preprocessing import PreprocessHumanID
from downstream.real_world.rw_train import HumanIDTrainer


def get_args():
    parser = argparse.ArgumentParser(description='Multimodal_PhyFM_on_Quality Downstream Stage.')

    # -------------------------------- Downstream Group--------------------------------
    args = parser.add_argument_group('Downstream Tasks.')
    # Added 'bidmc' to choices
    args.add_argument('--dataset_name', type=str, help='dataset name',
                      choices=['bp', 'vtac', 'stanford', 'dalia', 'bidmc', 'butppg', 'wesad', 'ppgbp', 'sdb', 'realworld'])
    args.add_argument('--stage', type=str, help='stage name', choices=['preprocessing', 'training'])
    args.add_argument('--raw_data_path', type=str, help='dataset input paths')
    args.add_argument('--seg_save_path', type=str, help='dataset save paths')

    # Task specific args
    args.add_argument('--task_name', type=str, default='af',
                      help='Specific task name for stanford: "af" or "quality"')

    # DALIA specific args
    args.add_argument('--test_subject', type=str, default='S15',
                      help='For DALIA LOSO training: specify the test subject (e.g., S15)')

    parser.add_argument('--backbone', type=str, help='The architecture of the feature extractor')
    args.add_argument('--sampling_rate', type=int, default=50, help='sampling rate')
    args.add_argument('--powerline_frequency', type=int, default=60, help='sampling rate for vtac')
    parser.add_argument('--out_dim', type=int, default=512, help='Output feature dimension.')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--model_save_path', type=str, default="model_saved",
                        help='Path to the directory where trained models will be saved.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of training.')
    parser.add_argument('--rsfreq', type=int, default=50, help='resampling rate (Hz)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    args.add_argument('--warmup_epochs', type=int, default=5, help='warm up epochs')

    args.add_argument('--checkpoint_path', type=str, default=None,
                      help='Path to the pre-trained checkpoint (required if pretrained is used).')

    args.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                      help='If specified, do NOT load pretrained weights (train from scratch).')

    args.add_argument('--freeze_backbone', action='store_true',
                      help='If specified, freeze the backbone weights and only train the head.')

    args.add_argument('--balance_strategy', type=str, default='focal_loss',
                      choices=['none', 'class_weights', 'focal_loss', 'oversample'],
                      help='Class imbalance handling strategy for classification tasks (default: focal_loss)')

    args.add_argument('--focal_gamma', type=float, default=2.0,
                      help='Gamma parameter for Focal Loss (default: 2.0, higher = more focus on hard examples)')

    args.add_argument('--label_smoothing', type=float, default=0.0,
                      help='Label smoothing for CrossEntropyLoss (default: 0.0, range: 0.0-0.2)')

    args.add_argument('--patience', type=int, default=15,
                      help='Patience for early stopping (default: 15 epochs)')

    args.add_argument('--unfreeze_after_epoch', type=int, default=-1,
                      help='Epoch to unfreeze backbone (-1 = never unfreeze, 0 = always unfrozen)')

    args.add_argument('--clip_grad', type=float, default=1.0,
                      help='Gradient clipping value (default: 1.0, 0 = no clipping)')

    args.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers (default: 4)')

    args.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to use for training (default: cuda)')

    args.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                      help='Optimizer (default: "adamw")')
    args.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                      help='Optimizer Epsilon (default: 1e-8)')
    args.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                      help='Optimizer Betas (default: None, use opt default)')
    args.add_argument('--momentum', type=float, default=0.9, metavar='M',
                      help='Optimizer momentum (default: 0.9)')
    args.add_argument('--weight-decay', type=float, default=0.05,
                      help='weight decay (default: 0.05)')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.dataset_name == "stanford":
        if args.stage == "preprocessing":
            processor = PreprocessStanford(args)
            processor.preprocess_save()
        elif args.stage == "training":
            trainer = StanfordTrainer(args)
            trainer.training()

    elif args.dataset_name == "dalia":
        print("🚀 Executing DALIA Task...")
        if args.stage == "preprocessing":
            processor = PreprocessDalia(args)
            processor.preprocess_save()
        elif args.stage == "training":
            trainer = DaliaTrainer(args)
            trainer.training()

    elif args.dataset_name == "bidmc":
        if args.stage == "preprocessing":
            processor = PreprocessBIDMC(args)
            processor.preprocess_save()
        elif args.stage == "training":
            trainer = BIDMCTrainer(args)
            trainer.training()

    elif args.dataset_name == "butppg":
        if args.stage == "preprocessing":
            processor = PreprocessButPPG(args)
            processor.preprocess_save()
        elif args.stage == "training":
            trainer = ButPPGTrainer(args)
            trainer.training()

    elif args.dataset_name == "wesad":
        if args.stage == "preprocessing":
            processor = PreprocessWESAD(args)
            processor.preprocess_save()
        elif args.stage == "training":
            trainer = WESADTrainer(args)
            trainer.train()

    elif args.dataset_name == "ppgbp":
        if args.stage == "preprocessing":
            processor = PreprocessPPGBP(args)
            processor.preprocess_save()
        elif args.stage == "training":
            trainer = PPGBPTrainer(args)
            trainer.training()

    elif args.dataset_name == "sdb":
        if args.stage == "preprocessing":
            processor = PreprocessSDB(args)
            processor.preprocess_save()
        elif args.stage == "training":
            trainer = SDBTrainer(args)
            trainer.training()

    elif args.dataset_name == "realworld":
        if args.stage == "preprocessing":
            processor = PreprocessHumanID(args)
            processor.preprocess_and_save()
        elif args.stage == "training":
            trainer = HumanIDTrainer(args)
            trainer.training()




