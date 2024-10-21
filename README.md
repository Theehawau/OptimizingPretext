SVM
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
| PPT Config | Model | Dataset | PTT Dataset Perc. | PPT Accuracy/Loss | Linear Probe | Full FT | 
|---|---|---|---|---|---|---|
| Random Init | ResNet50 | -- | -- | -- | 9.5 | -- |
| Contrastive | ResNet50 | TinyImageNet |  -- | -- | -- | -- |
| Contrastive | ResNet50 | Imagenet1k |  -- | -- | -- | -- |


### Dataset
Using all labels
Pretraining - 30% of Imagenet

Finetuning - 10% of Imagenet
Finetuning - full Imagenet

- logistic regression from scikit learn
- mlp
- Full finetuning

Using TinyImageNet for all experiments



