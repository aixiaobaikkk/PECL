## PECL

Code for this paper:Pseudo-label enhanced consistency learning for semi-supervised medical image segmentation
	

## Requirements

1. Create conda environment:

   ```bash
   conda create -n PECL python=3.10
   ```

2. Clone the repo:

   ```bash
   git clone https://github.com/aixiaobaikkk/PECL.git
   ```

3. Activate the environment:

   ```bash
   conda activate PECL
   ```

4. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### LA dataset

```bash
python train_LA.py
```

### ACDC dataset

```bash
python train_ACDC.py
```

### Pancreas dataset

```bash
python train_PA.py
```

## Acknowledgement

* This code is adapted from [UA-MT](https://github.com/yulequan/UA-MT), [DTC](https://github.com/HiLab-git/DTC.git) and [UniMatch](https://github.com/LiheYoung/UniMatch/tree/main/more-scenarios/medical) . .
