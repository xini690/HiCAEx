## HiCAEx: A Hierarchical Cross-Attention Expert Framework for Multimodal Retinal Disease Classification imaging

KeyWords: Retinal disease classification, Multimodal learning, Vision Transformer, Feature fusion, Multimodal retinal imaging.
### 🔧Install environment
1. Create environment with conda:
```
conda create -n HiCAEx_env python=3.11.0 -y
conda activate HiCAEx_env
```

2. Install dependencies
```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

3. Download Weight
```
Please download RETFound weights (https://github.com/rmaphoh/RETFound_MAE) and place them in the /weight directory
/weight/RETFound_cfp_weights.pth
/weight/RETFound_oct_weights.pth
```
### 📝DataSet
Organise your data into this directory structure
```
├── data folder
    ├──train
        ├──class_a
        ├──class_b
        ├──class_c
    ├──val
        ├──class_a
        ├──class_b
        ├──class_c
    ├──test
        ├──class_a
        ├──class_b
        ├──class_c
``` 

### 🛠️Training parameters
You can set the training parameters in the mtf_eye.yaml file
```
/sh/mtf_eye.yaml

epochs: 50
batch_size: 32 #32
blr: 5e-4
layer_decay: 0.65
...
```

### 🌱Run the code
```
cd /sh
sh target.sh
```

