<div align="center">

<h1> A Study on Optimal Pretext Tasks Configurations </h1>

<i> This project does not introduce a novel approach. </i> This work examines 3 Pre-Text tasks (Contrastive, Rotation, Jigsaw) individually and in sequential combinations under fixed computational constraints.

<div>
    <a href='https://www.linkedin.com/in/amirbek-djanibekov-a7788b201/' target='_blank'>Emilio Villa Cueva<sup> </a>&emsp;
    <a href='https://www.linkedin.com/in/toyinhawau/'> Hawau Olamide Toyin <sup></sup> </a>&emsp;
    <a href='https://www.linkedin.com/in/ajinkya-kulkarni-32b80a130/' target='_blank'>Karima Kadaoui<sup></a>&emsp;
</div>

<br>

</div>


## Environment & Installation

Python version: 3.10+

Clone this repo and install dependencies:
```bash
git clone https://github.com/Theehawau/OptimizingPretext.git
cd OptimizingPretext
pip install -r optimizepretext/requirements.txt
```


## Pre-training

For each task in (contrastive, jigsaw, rotation):

```bash
export task=contrastive
bash notebooks/$task/pretrain.sh
```

### Jigsaw
The parameters have to be set in the config file in jigsaw/configs; the hyperparameters and seed in the config file are the default ones.
To run a training session for jigsaw, we run the following:
```python run_jigsaw.py --config configs/config_jigsaw.yml```
In the config file the user must determine the path for saving the checkpoints.

## Fine-tuning

## Linear Probing





<!-- SVM
| PPT Config | Model | Dataset | PTT Dataset Perc. | PPT Accuracy | Linear Probe K=10| Linear Probe K=50| Linear Probe K=100| Full FT | 
|---|---|---|---|---|---|---|---|---|
| Random Init | ResNet50 | ImageNet | 0.1 | -- | 44.60% | 20.56% | | |
| Random Init | VIT 16 | ImageNet | 0.3 | -- | 47.40% | 34.24%
| Rotation | ResNet50 | ImageNet | 0.1 | 71.87% |  | |  | |
| Rotation | ResNet50 | ImageNet | 0.3 | 76.85% | 68.40% | 39.52% | | |
| Rotation | VIT 16 | ImageNet | 0.3 | 57.76% | 47.40% | | | |


Linear Layer
| PPT Config | Model | Dataset | PTT Dataset Perc. | PPT Accuracy | Linear Probe K=10| Linear Probe K=50| Linear Probe K=100| Full FT | 
|---|---|---|---|---|---|---|---|---|
| Random Init | ResNet50 | TinyImageNet | -- | -- | 14.80% | 4.68% | 3.68% | |
| Random Init | ResNet50 | ImageNet 1K | -- | -- | 20.40% | 4.85% | -- | |
| Rotation | ResNet50 | TinyImageNet | 1 | 74.53% | 31.9% | 13.28% | 6.24-running% | |
| Rotation | ResNet50 | ImageNet 1K | 0.3 | 76.85% | 48.20% | 22.7% | -- | |


LR
| PPT Config | Model | Dataset | PTT Dataset Perc. | FT Dataset Perc. | PPT Accuracy/Loss | Linear Probe | Full FT | 
|---|---|---|---|---|---|---|---|
| Random Init | ResNet50 | TinyImageNet | -- | 1 | -- | 9.5 | -- |
| Random Init | ResNet50 | Imagenet1k | -- | 0.1 | -- | 1.1 | -- |
| Contrastive | ResNet50 | TinyImageNet |  -- | -- | -- | 36.39 | 48.60 |
| Contrastive | ResNet50 | Imagenet1k |  0.3 |  0.1 | -- | 17.3 | -- | -- |
| Random Init KK | ResNet50 | TinyImageNet | -- | 1 | -- | 10.46 | -- |
| Random Init KK | ResNet50 | Imagenet1k | -- | 0.1 | -- | 1.96 | -- |
| Random Init KK | ResNet50 | Imagenet1k | -- | 0.3 | -- | 3.82 | -- |
| Rotation | ResNet50 | TinyImageNet | 1 | 1 | 74.53 | 11.67 | -- |
| Rotation | ResNet50 | Imagenet1k | 0.3 | 0.1 | 76.85 | 6.98 | -- |
| Rotation | ResNet50 | Imagenet1k | 0.3 | 0.3 | 76.85 | 7.41 | -- | -->



### Dataset
Using all labels
Pretraining - 30% of Imagenet

Finetuning - 10% of Imagenet
Finetuning - full Imagenet

- logistic regression from scikit learn
- mlp
- Full finetuning

Using TinyImageNet for all experiments



