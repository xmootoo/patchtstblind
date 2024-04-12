import os
import argparse
from patchtstblind.exp import Experiment
from datetime import datetime
from patchtstblind.utils.utils import tuple_type
from dotenv import load_dotenv
load_dotenv()
# Get the current date and time
now = datetime.now()

# Create the parser
parser = argparse.ArgumentParser(description="Run the Experiment")

# Dataloading (from: https://github.com/yuqinie98/PatchTST)
parser.add_argument("--data", type=str, default="custom", help="Name of the dataset class. Options: 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'custom'")
parser.add_argument('--root_path', type=str, default="./data/", help='Data directory containing all .csv files for each dataset')
parser.add_argument('--data_path', type=str, default="electricity.csv", help="Name of the dataset file")

# Default arguments (do not touch)
parser.add_argument('--features', type=str, default='M',
                        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate")
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

# Early Stopping
parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
parser.add_argument("--verbose", type=bool, default=True, help="Verbose print for Earlystopping class")
parser.add_argument("--delta", type=float, default=0., help="delta addative to the best scores within the EarlyStopping class")


# Scheduler (Supervised)
parser.add_argument("--warmup_steps", type=int, default=15, help="Number of warmup epochs for the scheduler")
parser.add_argument("--start_lr", type=float, default=1e-4, help="Starting learning rate for the scheduler for warmup")
parser.add_argument("--ref_lr", type=float, default=1e-3, help="End learning rate for the scheduler after warmup")
parser.add_argument("--final_lr", type=float, default=1e-6, help="Final learning rate by the end of the schedule (starting from ref_lr)")
parser.add_argument("--T_max", type=int, default=100, help="Maximum number of epochs for the scheduler")
parser.add_argument("--last_epoch", type=int, default=-1, help="Last epoch for the scheduler")


# Supervised Learning
parser.add_argument("--optim_type", type=str, default="adam", help="Optimizer for supervised learning: 'adam' or 'adamw'")
parser.add_argument("--criterion", type=str, default="MSE", help="Criterion for supervised learning: 'MSE' or 'SmoothL1'")
parser.add_argument("--seed", type=int, default=2024, help="Random seed")
parser.add_argument("--model_id", type=str, default="PatchTSTBlind", help="Model ID. Options: 'PatchTSTOG', 'PatchTSTBlind'")
parser.add_argument("--sample_sizes", type=tuple_type, default=(0.2,0.4, 0.4), help="Proportion of each sample (from 0 to 1) for train, validation, and test loaders.")
parser.add_argument("--seq_len", type=int, default=512, help="Sequence length of the input.")
parser.add_argument("--pred_len", type=int, default=96, help="Prediction length of the forecast window.")
parser.add_argument("--label_len", type=int, default=48, help="Overlap between the input and forecast windows.")
parser.add_argument("--learning_type", type=str, default="sl", help="Type of learning: 'sl' or 'ssl'")
parser.add_argument("--global_prenorm", type=bool, default=False, help="Whether to apply global prenormalization")
parser.add_argument("--train_split", type=float, default=0.6, help="Portion of data to use for training")
parser.add_argument("--test_split", type=float, default=0.2, help="Portion of data to use for testing")
parser.add_argument("--sl_batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--num_workers", type=int, default=64, help="Number of workers for data loading")
parser.add_argument("--num_enc_layers", type=int, default=3, help="Number of encoder layers in the model")
parser.add_argument("--d_model", type=int, default=128, help="Dimension of the model")
parser.add_argument("--d_ff", type=int, default=256, help="Dimension of the feedforward network model")
parser.add_argument("--num_heads", type=int, default=16, help="Number of heads in each MultiheadAttention block")
parser.add_argument("--num_channels", type=int, default=321, help="Number of time series channels")
parser.add_argument("--attn_dropout", type=float, default=0.2, help="Dropout value for attention")
parser.add_argument("--ff_dropout", type=float, default=0.2, help="Dropout value for feed forward")
parser.add_argument("--pred_dropout", type=float, default=0.1, help="Dropout value for prediction")
parser.add_argument("--batch_first", type=bool, default=True, help="Whether the first dimension is batch")
parser.add_argument("--norm_mode", type=str, default="batch1d", help="Normalization mode: 'batch1d', 'batch2d', or 'layer'")
parser.add_argument("--revin", type=bool, default=True, help="Whether to use instance normalization with RevIN.")
parser.add_argument("--revout", type=bool, default=True, help="Whether to use add mean and std back after forecast.")
parser.add_argument("--revin_affine", type=bool, default=True, help="Whether to use learnable affine parameters for RevIN.")
parser.add_argument("--eps_revin", type=float, default=1e-5, help="Epsilon value for reversible input")
parser.add_argument("--patch_dim", type=int, default=16, help="Patch dimension")
parser.add_argument("--stride", type=int, default=8, help="Stride value for patching")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay")
parser.add_argument("--sl_epochs", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--eta_min", type=float, default=0.00001, help="Minimum learning rate for CosineAnnealingLR")
parser.add_argument("--best_model_path", type=str, default="", help="Path to save the best model")
parser.add_argument("--acc", type=argparse.BooleanOptionalAction, help="Evaluate accuracy or not for classification tasks")
parser.add_argument("--api_token", type=str, default=os.environ.get('NEPTUNE_API_TOKEN', ''), help="Neptune API token")
parser.add_argument("--project_name", type=str, default="time-series-jepa/patchtst", help="Neptune project name")
parser.add_argument("--run_id", type=str, default=now.strftime("%Y-%m-%d_%H-%M-%S"), help="Neptune run ID")
parser.add_argument("--ddp", action=argparse.BooleanOptionalAction, help="Running distributive process on multiple nodes")
parser.add_argument("--scheduler", type=str, default="scheduler", help="Scheduler to use for learning rate annealing")
parser.add_argument("--master_port", type=str, default= "39545", help="Port for master node in SLURM. Check with 'netstat -tuln | grep <port>' for availability")
parser.add_argument('--early_stopping', type=bool, default=True, help='Whether to use early stopping')
parser.add_argument("--exp_id", type=str, default="test", help="Experiment ID, e.g. 'jepa_patchtstblind_512_96'")


# Parse the arguments
args = parser.parse_args()

if __name__=="__main__":
    exp = Experiment(args)
    exp.run()
