import os
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, RandomSampler, Subset
from sklearn.preprocessing import StandardScaler


# Exchange rate: num_timesteps = 7,588 num_channels = 8

def get_dataloaders(root_path : str,
                 dataset_file : str="electricity.csv",
                 size : Tuple[int, int, int]=None,
                 freq : Literal['m', 'h', 'd']='h',
                 features : Literal["all", "target"]="all",
                 learning_type : Literal["ssl", "sl"]="sl",
                 use_time_features : bool=True,
                 scale : bool=False,
                 target : str | None=None,
                 return_type : Literal["numpy", "tensor"]="tensor",
                 train_split=0.7,
                 test_split=0.2,
                 batch_size=32,
                 num_workers=4,
                 shuffle=True,
                 sample_sizes: Tuple=(None, None, None)):
    """
    Returns training, validation, and test DataLoader objects.

    Args:
        root_path: The root directory of the dataset.
        dataset_file: The name of the dataset file.
        size: The sequence length, window stride length, and prediction length.
        freq: The frequency of the time series data.
        features: The features to include in the dataset.
        learning_type: The type of learning: 'sl' for supervised learning or 'ssl' for self-supervised learning.
        use_time_features: Whether to include time features in the dataset.
        scale: Whether to normalize the dataset using StandardScaler.
        target: The target column to forecast.
        return_type: The type of data to return: 'numpy' or 'tensor'.
        train_split: The proportion of the dataset to use for training.
        test_split: The proportion of the dataset to use for testing.
        batch_size: The batch size for the DataLoader.
        num_workers: The number of workers for data loading.
        shuffle: Whether to shuffle the DataLoader.
        sample_sizes: The sample sizes for the training, validation, and test sets.
    """

    datasets = get_datasets(root_path=root_path,
                           dataset_file=dataset_file,
                           size=size,
                           freq=freq,
                           features=features,
                           learning_type=learning_type,
                           use_time_features=use_time_features,
                           scale=scale,
                           target=target,
                           return_type=return_type,
                           train_split=train_split,
                           test_split=test_split)

    # (Optional) Sampling without replacement from datasets (e.g., downsampling for large datasets)
    if sample_sizes and all(size is not None for size in sample_sizes): # If every entry in sample_sizes is not None

        # Convert to number of samples from proportions
        if all(0 <= size < 1 for size in sample_sizes):
            num_samples = [int(len(datasets[i]) * sample_sizes[i]) for i in range(len(datasets))]
        else:
            num_samples = [int(size) for size in sample_sizes]
        
        # Assign indices for each sample
        indices = [list(RandomSampler(datasets[i], replacement=False, num_samples=num_samples[i])) for i in range(len(datasets))]
        
        # Get samples from each dataset
        samples = [Subset(datasets[i], indices[i]) for i in range(len(datasets))]
    else:
        samples = datasets

    # Load each sample into a DataLoaders
    dataloaders = []
    for sample in samples:
        dataloader = DataLoader(sample, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        dataloaders.append(dataloader)

    # Include length for each loader
    dataset_lengths = [len(sample) for sample in samples]
    dataloaders.append(dataset_lengths)

    return tuple(dataloaders)


def get_datasets(root_path : str,
                 dataset_file : str="electricity.csv",
                 size : Tuple[int, int, int]=None,
                 freq : Literal['m', 'h', 'd']='h',
                 features : Literal["all", "target"]="all",
                 learning_type : Literal["ssl", "sl"]="sl",
                 use_time_features : bool=True,
                 scale : bool=False,
                 target : str | None=None,
                 return_type : Literal["numpy", "tensor"]="tensor",
                 train_split : float=0.7,
                 test_split : float=0.2):
    """
    Returns training, validation, and test DatasetLTSF objects.
    """
    datasets = []
    for flag in ["train", "val", "test"]:
        datasets.append(DatasetLTSF(root_path=root_path,
                           dataset_file=dataset_file,
                           flag=flag,
                           size=size,
                           freq=freq,
                           learning_type=learning_type,
                           use_time_features=use_time_features,
                           scale=scale,
                           target=target,
                           return_type=return_type,
                           train_split=train_split,
                           test_split=test_split))

    return tuple(datasets)



def _torch(*args):
        """
        Converts numpy arrays to PyTorch tensors.

        Args:
            *args: One or more numpy arrays.

        Returns:
            A generator yielding PyTorch tensors corresponding to the input numpy arrays,
            converted to float data type.

        Example:
        ```
            import numpy as np
            import torch

            arr1 = np.array([1, 2, 3])
            arr2 = np.array([4, 5, 6])
            tensors = _torch(arr1, arr2)

            for tensor in tensors:
                print(tensor)
        ```
        """
        return tuple(torch.from_numpy(np_obj).float() for np_obj in args)

class DatasetLTSF(Dataset):
    """
    Dataset class for Long-Term Time Series Forecasting (LTSF) or Time Series classification tasks.

    Returns:
        Dataset: A Dataset class where (x, y, seq_x_time, seq_y_time) are returned.
        x (torch.Tensor): Input window of shape (num_channels, seq_len).
        y (torch.Tensor): Target window of shape (num_channels, pred_len).
        seq_x_time (torch.Tensor): Sequence of time indices for x.
        seq_y_time (torch.Tensor): Sequence of time indices for y.
    """


    def __init__(self,
                 root_path : str,
                 dataset_file : str="electricity.csv",
                 flag :Literal['train', 'test', 'val']="train",
                 size :Tuple[int, int, int]=None,
                 freq :Literal['m', 'h', 'd']='h',
                 features : Literal["all", "target"]="all",
                 learning_type : Literal["ssl", "sl"]="sl",
                 use_time_features : bool=True,
                 scale : bool=False,
                 target : str | None=None,
                 return_type : Literal["numpy", "tensor"]="tensor",
                 train_split : float=0.7,
                 test_split: float=0.2):
        flag = flag.lower()

        if size == None:
            self.seq_len = 24 * 4 * 4  # sequence length
            self.stride_len = 0        # stride away from label sequence x length
            self.pred_len = 24 * 4 * 2 # prediction length
        else:
            assert len(size) == 3, ""
            # NOTE add check for better determining seq_len, stride_len, pred_len
            self.seq_len = size[0]     # sequence length
            self.stride_len = size[1]  # stride away from label sequence x length
            self.pred_len = size[2]    # prediction length

        # match/switch statement that check what types of data were looking at
        match flag:
            # flag variable any of the present vales in case
            case 'train' :
                self.type = 0
            case 'val':
                self.type = 1
            case 'test':
                self.type = 2
            # default if none of the string above then raise a Value error...
            case _:
                raise ValueError(f"type {flag} does not exist. Only types are (train, test, & val)")

        self.freq = freq
        self.dataset_file = dataset_file
        self.root_path = root_path
        self.return_type = return_type
        self.scale = scale
        self.target = target
        self.features = features
        self.learning_type = learning_type
        self.use_time_features = use_time_features
        self.train_split=train_split
        self.test_split=test_split
        self.rd() # read csv folder of the existing dataset...

    def __len__(self):
        """
            Return length of the dataset not including labels
        """
        return len(self.x) - self.seq_len - self.pred_len + 1

    def rd(self):
        """
            Reads and processes the dataset file specified in the object's attributes.

            This function performs the following operations:
            - Checks if the root directory and dataset file exist.
            - Reads the CSV dataset file into a pandas DataFrame.
            - Depending on the 'features' attribute, selects either all columns or only the target column.
            - If scaling is enabled, scales the data using StandardScaler.
            - Extracts date-related features from the 'date' column.
            - Sets the 'x', 'y', and 'data_stamp' attributes for later use.

            Raises:
                IOError: If the root directory does not exist.
                AssertionError: If the dataset file is not a CSV file.
                FileNotFoundError: If the dataset file does not exist in the root directory.

            Example:
                # Assuming 'self' is an object with the necessary attributes set.
                self.rd()
        """
        if not os.path.isdir(self.root_path):
            raise IOError(f"root directory {self.root_path} does not exist, please try one that exist...")

        file = os.path.join(self.root_path, self.dataset_file)

        assert file.endswith(".csv"), f"{self.__class__.__name__} only supports csv files formats..."

        if not os.path.isfile(file):
            _suggested = [f for f in os.listdir(self.root_path) if f.endswith(".csv")]
            _suggested_stdout = f'please try ({", ".join(_suggested)})'\
                                 if len(_suggested) != 0 \
                                 else 'No other csv files exist in this folder'
            raise FileNotFoundError(f"""{file} does not exist within, {_suggested_stdout}""")

        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.dataset_file))
        if "ETTm" in self.dataset_file:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        elif "ETTh" in self.dataset_file:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        else:
            num_train = int(len(df_raw) * self.train_split)
            num_test = int(len(df_raw) * self.test_split)
            num_vali = len(df_raw) - num_train - num_test

            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.type]
        border2 = border2s[self.type]

        if self.features == 'all':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'target':
            assert self.target is not None, "can not select a target, and not choose a target"
            assert df_raw.columns[1:].isin([self.target]).any(), f"{self.target} is not in the column={df_raw.columns[1:]}"
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']].iloc[border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        df_stamp['month'] = df_stamp['date'].dt.month
        df_stamp['day'] = df_stamp['date'].dt.day
        df_stamp['weekday'] = df_stamp['date'].dt.weekday
        df_stamp['hour'] = df_stamp['date'].dt.hour
        if "ETT" in self.dataset_file:
            df_stamp['minute'] = df_stamp['date'].dt.minute // 15

        data_stamp = df_stamp.drop(['date'], axis=1).values


        self.x = data[border1:border2]
        self.y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        seq_start = idx
        seq_end = seq_start + self.seq_len
        r_start = seq_end + self.stride_len
        r_end = r_start + self.pred_len

        seq_x = self.x[seq_start:seq_end]
        seq_y = self.y[r_start:r_end]

        seq_x_steps = self.data_stamp[seq_start:seq_end]
        seq_y_steps = self.data_stamp[r_start:r_end]

        if self.learning_type == "sl":
            output = (seq_x.T, seq_y.T, seq_x_steps, seq_y_steps) \
                     if self.use_time_features\
                     else (seq_x, seq_y)
        else:
            output = (seq_x.T, seq_x_steps) \
                     if self.use_time_features\
                     else (seq_x)

        if self.return_type == "tensor":
            return _torch(*output)
        else:
            return output

    def __str__(self):
        return f"DatasetLTSF(root_path=\"{self.root_path}\", "\
               f"dataset=\"{self.dataset_file}\", "\
               f"size={self.__len__()}, "\
               f"t"\
               f"learning_type=\'{self.learning_type}\')"
