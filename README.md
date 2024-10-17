# Unlearning under adversary

### Requirements
Ensure the following packages are installed:

```
numpy==1.24.1
requests==2.32.3
torch==2.4.0
torchvision==0.19.0
pandas==1.5.3
Pillow==10.2.0
higher==0.2.1
tqdm==4.66.4
```


### Running the CIFAR10 Experiment

To run the unlearning experiment on the CIFAR10 dataset, use the following command:
```bash
cd cifar
python unlearning-cifar10.py --unlearn_n 20 --unlearn_epochs 1 --unlearn_lr 0.02 --optimize_images --log_folder ./log --step_optimized 1000
```



| Parameter                | Type    | Default Value | Description                                      |
|--------------------------|---------|---------------|--------------------------------------------------|
| `--unlearn_n`             | `int`   | 100           | Number of data points to unlearn.                |
| `--forget_set_seed`       | `int`   | -1            | Seed for the forget set selection.              |
| `--unlearn_epochs`        | `int`   | 1             | Number of epochs for the unlearning process.      |
| `--unlearn_lr`            | `float` | 0.02          | Learning rate for the unlearning method.       |
| `--adv_lr`                | `float` | 0.05          | Learning rate for the adversarial process.        |
| `--optimize_images`       | `flag`  | `False`       | Flag to optimize images during the process.       |
| `--save_images`           | `flag`  | `False`       | Flag to save optimized images.                    |
| `--step_optimized`        | `int`   | 5             | Number of steps for optimization.                 |
| `--clip_norm`             | `float` | 3.0           | Noise norm clipping value.                    |
| `--unlearn_method`        | `str`   | "ga"          | Method for unlearning (e.g., `ga`, `ga_klr`, `ga_gdr`,).  |
| `--log_folder`            | `str`   | "./log"       | Folder to store log files.                       |


### Black-Box Attack Parameters

| Parameter                | Type    | Default Value | Description                                      |
|--------------------------|---------|---------------|--------------------------------------------------|
| `--attack_method`         | `str`   | "white_box"   | Method of attack (`white_box` or `black_box`).    |
| `--p_size`                | `int`   | 1             | Size of the perturbation set.                    |
| `--m`                     | `int`   | 1             | Number of iterations in the attack.              |
| `--dis`                   | `float` | 0.1           | Distance metric threshold.                       |
| `--ensure_decrease`       | `flag`  | `False`       | Ensure loss decrease with the new gradiet estimator. |


### Citing
If you use this work, please cite our work as 😊:

@article{huang2024unlearn,
  title={Unlearn and Burn: Adversarial Machine Unlearning Requests Destroy Model Accuracy},<br>
  author={Huang, Yangsibo and Liu, Daogao and Chua, Lynn and Ghazi, Badih and Kamath, Pritish and Kumar, Ravi and Manurangsi, Pasin and Nasr, Milad and Sinha, Amer and Zhang, Chiyuan},<br>
  journal={arXiv preprint arXiv:2410.09591},<br>
  year={2024}
}