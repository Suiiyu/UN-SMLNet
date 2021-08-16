# UN-SMLNet
Code for our paper "Uncertainty Guided Symmetric Multi-Level Supervision Network for Left Atrium Segmentation on Late Gadolinium-Enhanced MRI". 

- Proposed an uncertainty guided objective function to refine the left atrium segmentation based on Jenson-Shannon (JS) discrepancy.
- Conducted an symmetric multi-level supervision network for multi-scale representation learning.

The pipeline of our method is show below:

We reconstructed the worst predicted case (a) and the best predicted case (b) of our UN-SMLNet and the corresponding reconstructed predictions of other models through 3D Slicer .

<p align="center">
    <img src="images/framework.png" width="700" height="400">
</p>


## Requirements

Python 3.5

Keras based on Tensorflow

## Data process

Our model was trained and evaluated on 100 LGE MRI volumes of AF patients, which was provided by 2018 Atrial Segmentation Challenge in the STACOM 2018. Each volume contained 88 slices along Z direction with a spatial dimension of either $576\times576$ or $640\times640$. The ground truth of LA in this dataset was performed by three trained observers. The ground truth included LA endocardial surface with the mitral valve, LA appendage, and an part of the pulmonary veins (PVs).

We firstly extracted 2D slices along the Z direction. And then, these slices were cropped into $288\times288$ around the center of the slices to omit most of the unrelated region. The 100 volumes were randomly divided into training (N=72), validation (N=8), and testing (N=20) sets. The pixel value of these slices was normalized into $[0,1]$ by min-max normalization. For enlarging the training set, online augmentation approaches for the training slices were adopted.

## Training

**Folder structure**:

```python
traindata_dir = 'data/train_volumes'
testdata_dir = 'data/test_volumes'
```

**Run**

```python
train: python train.py
```

Testing is on the 20 LGE CMR data. The testing data should be kept in the original status. 

```python
test: python test.py
```

After training the segmentation model, we reconstruct the prediction results in the original shape. Then, a connected component analysis is performed to remain the largest connected region as the final segmentation result. Our segmentation model is evaluated by the five evaluation metrics, which are **Dice score**, **Jaccard score**, **Average Symmetric Surface Distance (ASSD)**, and **95% Hausdorff distance (HD)**. 

## Results

The score of metrics during the test stage is shown in the box diagrams. 

<p align="center">
    <img src="images/box_compare_result.png" width="7000" height="200">
</p>


We reconstructed the worst predicted case (a) and the best predicted case (b) of our UN-SMLNet and the corresponding reconstructed predictions of other models through 3D Slicer .

<p align="center">
    <img src="images/3drecon.png" width="700" height="300">
</p>

