## Dependencies
- Python==3.9
- numpy==1.24.2
- scikit_learn==1.2.2
- torch==2.0.0
- tqdm==4.64.1
- Maybe other library version also works.

## Data preparation
The multi-model embedding of MMKGs are too large so you should download them from the [Google Drive Link](https://drive.google.com/file/d/1dKJdJunb11kDtFr5NLfPlFknS7cRdm9W/view?usp=sharing). Please unzip the embedding files and put them in the corresponding path in `datasets/`



## Train and Evaluation

Here is an example for DB15K dataset:

```bash
nohup python train.py --cuda 0 --lr 0.001 --mu 0.0001 --dim 200 --dataset MKG-W --epochs 2000 > log.txt &

nohup python train.py --cuda 1 --lr 0.0005 --mu 0.0001 --dim 300 --dataset MKG-Y --epochs 2000 > log.txt &
```
The evaluation results will be printed in the command line after training.

