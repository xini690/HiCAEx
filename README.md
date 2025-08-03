## HiCAEx: A Hierarchical Cross-Attention Expert Framework for Multimodal Retinal Disease Classification imaging

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
Please download RetFound weights and place them in the /weight directory
/weight/RETFound_cfp_weights.pth
/weight/RETFound_oct_weights.pth
```

4. Run the code 
```
cd /sh
sh target.sh
```

