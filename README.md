# PatchTST Model Implementation: Reconstructed Benchmarks (ICLR 2023)

### This is an un-offical implementation of PatchTST: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730). 

The original authors provide a video with a concise overview of the paper for individuals seeking further comprehension: https://www.youtube.com/watch?v=Z3-NrohddJw


## Key Designs

:star2: **Patching**: segmentation of time series into subseries-level patches which are used as input tokens to the Transformer.

:star2: **Channel-independence**: each channel is processed independently by the Transformer to predict the forecast. The prediction for multivariate time series is a concatenation of the channel predictions, as shown in the figure below.

![alt text](https://github.com/xmootoo/patchtstblind/blob/main/assets/model.png)

## Acknowledgement

The original authors acknowledgement these repos, and so in turn we appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai


## Get Started

### Download Datasets

To download the datasets, please use the following link to access our shared Google Drive folder. Once downloaded, place the `data` folder in root of the github directory

[https://drive.google.com/drive/folders/14VMQ5msUCNvZkEJqpEfvF2Ul0iBpzWTz?usp=sharing](https://drive.google.com/drive/folders/14VMQ5msUCNvZkEJqpEfvF2Ul0iBpzWTz?usp=sharing)

### Install the Package
Before using the code, please install `patchtstblind` as a package to allow imports using the following:
```bash
pip install -e .
```

### Train Model

To run the experiments, navigate to the `patchtstblind/jobs/local` directory:
```bash
cd patchtstblind/jobs/local
```
and locate the `submit.py` file. Execute the file using the following command:
```bash
python submit.py --exp_name "<exp_name>"
```
All experiments are contained in `patchtstblind/jobs/exp` with experimental parameters contained in the `args.yaml` files. For example, if you wish to run the ETTh1 experiment with sequence length $L = 512$ and prediction length $T = 96$:
```bash
python submit.py --exp_name "etth1/etth1_512_96"
```

### Citation 

```
@inproceedings{Yuqietal-2023-PatchTST,
  title     = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author    = {Nie, Yuqi and
               H. Nguyen, Nam and
               Sinthong, Phanwadee and 
               Kalagnanam, Jayant},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```
