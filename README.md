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


## Pre-text Training

For each pre-text task training, follow instructions below:

### Jigsaw
Default training configurations are in [jigsaw/configs](jigsaw/configs) 
```bash
python jigsaw/run_jigsaw.py --config jigsaw/configs/config_jigsaw.yml
```
In the config file the user must update the path to save the checkpoints.

### Rotation
Default training configurations are in [rotation/configs](rotation/configs) 
```bash
python rotation/run_rotation.py --config rotation/configs/config_rotation.yml
```
In the config file the user must update the path to save the checkpoints.

### SimCLR
Default training configurations are in [contrastive/config.py](./contrastive/config.py)
```bash
python contrastive/pretrain.py --config pretrain --train
```

## Fine-tuning

To finetune the full resnet50 on either of our reported dataset, update the config and set the path to the pre-text trained weights checkpoint. 

```
# Training
python contrastive/finetune.py --config "full_ft" --data "tinyimagenet" -t 
#other data choices: caltech, voc2007

# Evaluating
python contrastive/finetune.py --config "full_ft" --data "tinyimagenet" 
```

To finetune on new dataset, update hyperparameters in [contrastive/config.py](./contrastive/config.py)

## Linear Probing

To evaluate PTT using linear probing 

```bash 
python linear_probe_example/linear_probe.py
```

## Checkpoints
Our trained checkpoints can be found [here](https://drive.google.com/drive/folders/1uzG_xTdJE9v9W20U5Jt2uiI7U1h1BeDP?usp=sharing) along with the teaser video and the slides.



