# Robust Motion Inbetweening

## Credit
- [Research paper by Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, Christopher Pal](https://arxiv.org/abs/2102.04942)
- [Ubisoft La Forge Animation Dataset ("LAFAN1")](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
- [Original implemention by xjwxjw](https://github.com/xjwxjw/Pytorch-Robust-Motion-In-betweening)

## How to run this
1. Load dataset
    - Download the dataset repo into the root directory from [here](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
    - Extract the lafan1/lafan.zip file and move the ubisoft-laforge-animation-dataset/lafan1 folder to the root directory
    - You may want to create a lafan1_small folder in this directory with only 1 activity for each subject for faster debuggin
    - Run the ExploreDataset notebook to see if the dataset was loaded properly

2. Preprocessing
    - Run the flip_bvh.py file by passing it the path to the lafan1 and lafan1_small folders.

3. Training
    - Use the Train notebook to train the generator/discriminator.

4. Evaluation
    - Use the Evaluation notebook to generate frames and save video