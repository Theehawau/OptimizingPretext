SVM
| PPT Config | Model | Dataset | PTT Dataset Perc. | PPT Accuracy | Linear Probe K=10| Linear Probe K=50| Linear Probe K=100| Full FT | 
|---|---|---|---|---|---|---|---|---|
| Random Init | ResNet50 | ImageNet | 0.1 | -- | 0.4460 | 0.2056 | | |
| Random Init | VIT 16 | ImageNet | 0.3 | -- | 0.4740 | 0.3424
| Rotation | ResNet50 | ImageNet | 0.1 | 71.87% |  | |  | |
| Rotation | ResNet50 | ImageNet | 0.3 | 76.85% | 0.6840 | 0.3952 | | |
| Rotation | VIT 16 | ImageNet | 0.3 | 57.76% | 0.4740 | | | |


Linear Layer
| PPT Config | Model | Dataset | PTT Dataset Perc. | PPT Accuracy | Linear Probe K=10| Linear Probe K=50| Linear Probe K=100| Full FT | 
|---|---|---|---|---|---|---|---|---|
| Random Init | ResNet50 | TinyImageNet | -- | -- | 14.80% | 4.68% | 3.68% | |
| Random Init | ResNet50 | ImageNet 1K | -- | -- | 20.40% | 4.85% | -- | |
| Rotation | ResNet50 | TinyImageNet | 1 | 74.53% | 31.9% | 13.28% | 6.24-running% | |
| Rotation | ResNet50 | ImageNet 1K | 0.3 | 76.85% | 48.20% | 22.7% | -- | |


