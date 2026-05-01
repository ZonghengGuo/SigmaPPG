import argparse
from preprocessing import processor


def get_args():
    parser = argparse.ArgumentParser(description='Diffusion FM Preprocessing Stage')

    # -------------------------------- Preprocessing Group--------------------------------
    preprocess_args = parser.add_argument_group('Data preprocessing parameters')
    preprocess_args.add_argument('--dataset_name', type=str, help='name of dataset')
    preprocess_args.add_argument('--raw_data_path', type=str, help='list of dataset input paths')
    preprocess_args.add_argument('--seg_save_path', type=str, help='where to save segmented data')
    preprocess_args.add_argument('--rsfreq', type=int, default=50, help='resampling rate (Hz)')
    preprocess_args.add_argument('--h5_file_numbers', type=int, default=50000, help='segment numbers in each h5 file')
    preprocess_args.add_argument('--window_length', type=int, default=240, help='window time length in seconds (default: 240s)')
    preprocess_args.add_argument('--patch_length', type=float, default=1.0, help='patch time length in seconds (default: 1.0s)')


    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    print("Start processing dataset")

    if args.dataset_name == 'mimic_iii' or args.dataset_name == 'mimic_iv':
        database = processor.MimicProcessor(args)
        database.run_processing()

    elif args.dataset_name == 'vitaldb':
        database = processor.VitaldbProcessor(args)
        database.run_processing()