# Reconstructed PatchTST Benchmarks (ICLR 2023)

### This is an un-offical implementation of PatchTST: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730). 

Original workers on the paper offer a video that provides a concise overview of our paper for individuals seeking a rapid comprehension of its contents: https://www.youtube.com/watch?v=Z3-NrohddJw


## Key Designs

:star2: **Patching**: segmentation of time series into subseries-level patches which are served as input tokens to Transformer.

:star2: **Channel-independence**: each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series.

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


### Get Started

## Download Datasets

To download the datasets, please use the following link to access our shared Google Drive folder. Once downloaded, place the data folder in root of the github directory

[https://drive.google.com/drive/folders/14VMQ5msUCNvZkEJqpEfvF2Ul0iBpzWTz?usp=sharing](https://drive.google.com/drive/folders/14VMQ5msUCNvZkEJqpEfvF2Ul0iBpzWTz?usp=sharing)

To run tests on the model, navigate to the `patchtstblind/jobs/local` directory and locate the `submit.py` file. Execute the file using the following command:

there you can run ``python submit.py --exp_name etth1/etth1_512_96`` running that argument yaml (args.yaml) within the ``patchtstblind/jobs/exp/patchtst/etth1/etth1_512_96`` directory.

```bash
cd patchtstblind/jobs/local
python submit.py --exp_name etth1/etth1_512_96
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