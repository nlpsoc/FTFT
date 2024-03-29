# FTFT

This repository contains the implementation for the paper 
"[FTFT: efficient and robust Fine-Tuning by transFerring Training dynamics](https://arxiv.org/abs/2310.06588)".


## Abstract
Despite the massive success of fine-tuning Pre-trained Language Models (PLMs), they remain susceptible to out-of-distribution input. Dataset cartography is a simple yet effective dual-model approach that improves the robustness of fine-tuned PLMs. It involves fine-tuning a model on the original training set (i.e. reference model), selecting a subset of important training instances based on the training dynamics, and fine-tuning again only on these selected examples (i.e. main model). However, this approach requires fine-tuning the same model twice, which is computationally expensive for large PLMs. In this paper, we show that 
1) Training dynamics are highly transferable across model sizes 
and pre-training methods, and that
2) Fine-tuning main models using these selected training instances 
achieves higher training efficiency than empirical risk minimization (ERM).

Building on these observations, we propose a novel fine-tuning approach: Fine-Tuning by transFerring Training dynamics (FTFT). Compared with dataset cartography, FTFT uses more efficient reference models and aggressive early stopping.  FTFT achieves robustness improvements over ERM while lowering the training cost by up to $\sim 50\%$.



## Get Started

### Install dependencies
```bash
python -m venv ftft_venv
source ftft_venv/bin/activate
pip install -r requirements.txt
```

### Configure HuggingFace Accelerate

We offer an example of configuration file
for Huggingface Accelerate in the `accelerate_config` folder, 
using one single GPU and `bfloat16` mixed precision training. 
For more customization, please refer to 
the [official guide](https://huggingface.co/docs/accelerate/basic_tutorials/install#configuring--accelerate). 


### Configure Weights & Biases

If you would like to use Weights & Biases to track your experiments, you need to configure it first.
Follow the [official guide](https://docs.wandb.ai/quickstart) for configuration.

### Prepare Data

Download data from [Google drive](https://drive.google.com/file/d/1MWaJo8rBDaX2I286B3NHWoZATlDhJjq1/view?usp=sharing), 
and decompress it in the root folder of this repository. 
The data folder should contain the following folders:
```
datasets/
├── hsd
│   ├── cad
│   └── dynahate
└── nli
    ├── anli
    │   ├── R1
    │   ├── R2
    │   └── R3
    └── mnli
```

## Run Experiments

Experiments can be constructed by ```run.py``` and YAML configuration files,
in which you can easily specify the ```Accelerate``` configuration, 
random seeds, reference and main models, number of training steps, 
wandb configuration, and other hyperparameters.

We offer example configuration files to reproduce our experiments of 
using different sizes of ```DeBERTaV3``` as reference models to fine-tune ```DeBERTaV3-Large``` 
on both NLI and HSD tasks. 
For example, you can run 
```bash
python run.py --config_path run_config/nli_erm_debertav3_base.yaml
python run.py --config_path run_config/nli_dm_debertav3_base_to_debertav3_large.yaml
```
to obtain the bash scripts for these experiments. 
By default, the scripts will be saved in the same folder under the same name as the configuration file, 
with the extension changed to ```.sh```.


## Citation

If you find this repository useful, please cite our paper

```bibtex
@article{du2023ftft,
  title={FTFT: efficient and robust Fine-Tuning by transFerring Training dynamics},
  author={Du, Yupei and Gatt, Albert and Nguyen, Dong}, 
  journal={arXiv preprint arXiv:2310.06588},
  url={https://arxiv.org/abs/2310.06588},
  year={2023}
}
```