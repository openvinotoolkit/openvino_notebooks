from argparse import ArgumentParser


def get_arguments_parser():
    parser = ArgumentParser()

    parser.add_argument('-c', '--nncf_config', help='Path to an NNCF config file')

    parser.add_argument(
        "--mode",
        "-m",
        nargs='+',
        choices=['train', 'test', 'export'],
        default='train',
        help=("train: performs training and validation; test: tests the model"
              "on the validation split of \"--dataset\"; export: exports the model to .onnx"))

    model_init_mode = parser.add_mutually_exclusive_group()
    model_init_mode.add_argument(
        "--resume",
        metavar='PATH',
        type=str,
        default=None,
        dest='resuming_checkpoint_path',
        help="Specifies the .pth file with the saved model to be tested (for \"-m test\""
             "or to be resumed from (for \"-m train\"). The model architecture should "
             "correspond to what is specified in the config file, and the checkpoint file"
             "must have all necessary optimizer/compression algorithm/metric states required.")

    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        metavar='N',
        help="Batch size. Will be split equally between multiple GPUs in the "
             "--multiprocessing-distributed mode."
             "Default: 10")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs. Default: 300")

    # Dataset
    parser.add_argument(
        "--data",
        dest="dataset_dir",
        type=str,
        help="Path to the root directory of the selected dataset. ")

    parser.add_argument('--to-onnx', type=str, metavar='PATH', default=None,
                        help='Export to ONNX model by given path')

    return parser
