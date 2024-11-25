# Data collection and preprocessing

To ensure reproducibility of the results, coordinates of the tiles used in the paper (necessary to extract tile images and features from whole-slide images) can be downloaded from https://drive.google.com/file/d/1PJsUv1SQieJs7hqtWOqW68v1K9c-mIF6/view?usp=sharing.
Activity signature values can be found in `assets/activity_signature` and the python formula is in `he2tead/preprocessing/signature_computation.py`

# Install

To install openslide, do:
```bash
apt-get update -qq && apt-get install openslide-tools libgeos-dev -y 2>&1
```

Then to install he2tead:
```bash
pip install -e .
```

# Dataset
The general class to load data is TCGADataset present in `he2tead/data/dataset.py`.

# Feature Extraction
## ResNet50 pre-trained with supervised learning on ImageNet dataset
In the file `he2tead/preprocessing/imagenet_extraction.py`, we provide an example on how to extract features
from each tiles, given their coordinates, using a ResNet50 pre-trained on ImageNet dataset.

## Wide ResNet50 x2 pre-training with MoCo-v2 on histology images
The feature extractor used in our study is a Wide ResNet50 x2, that was pre-trained with MoCo v2 on 4 million tiles from 
TCGA-COAD dataset. 

The code to train such model is available here: https://github.com/facebookresearch/moco.

Other feature extractors tailored from histologyn such as https://github.com/owkin/HistoSSLscaling have been open sourced in recent years.

# Predictive models training

The script to train models to predict the TEAD_500 signature in a cross-validated fashion is available here: `he2tead/engine/run_cv.py`.

An example is provided as a bash script
```bash
cd he2tead/engine
bash run.sh
```
