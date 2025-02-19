# DesmokeVIT: Desmoking with CycleGAN-Enhanced Vision Transformer for Unpaired Laparoscopic Images

## Background and Motivation

Robot-Assisted Surgery (RAS) has made significant strides in recent years, enhancing the precision of surgical procedures while minimizing invasiveness and improving patient outcomes. However, one major challenge remains: smoke generated during surgery, which obstructs the surgical field and hampers visibility. This issue is particularly prominent in laparoscopic surgery, where the removal of smoke during procedures is essential for clear visualization.

Existing approaches for laparoscopic smoke removal have shown some progress, but they often fall short in handling the unique challenges posed by laparoscopic images. Specifically, these methods struggle with incomplete smoke removal, especially when the red bias phenomenon is present in the images. To address these limitations, we introduce **DesmokeVIT**, a novel approach to smoke removal that leverages the power of attention mechanisms to capture intricate image features. Built upon the CycleGAN framework for unpaired image-to-image translation and enhanced with Vision Transformer modules, **DesmokeVIT** aims to improve smoke removal effectiveness while preserving important visual details in the surgical images.

Our comprehensive evaluations on related datasets demonstrate that DesmokeVIT outperforms existing methods, achieving state-of-the-art performance across multiple automated metrics, all while maintaining a relatively low parameter count. This makes our approach both efficient and effective for real-world surgical applications, enabling clearer and more reliable laparoscopic imagery.

## Installation and Usage

To use **DesmokeVIT**, follow the steps below for installation and setup:

### Prerequisites
Make sure you have the following installed:
- Python 3.9+
- PyTorch (with CUDA support for GPU acceleration)
- Other dependencies are listed in the `requirements.txt` file.

### Installation
Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/DesmokeVIT.git
cd DesmokeVIT
pip install -r requirements.txt
```

## Dataset Download

You can download the datasets used for training and testing **DesmokeVIT** from the following links:

- **DesmokeLAP Dataset**: [Download from UCL](https://www.ucl.ac.uk/interventional-surgical-sciences/weiss-open-research/weiss-open-data-server/desmoke-lap)
- **DesmokeData Dataset**: [Download from GitHub](https://github.com/wxia43/DesmokeData)

Please follow the provided instructions on each page to download and prepare the dataset for use.

### Training

To train the model, use the following command sample:

```
python train.py --dataroot ./datasets/hazy2clear_0206 --name vit_512_100epoch_vgg --model cycle_gan --batch_size 4 --netG vit --n_epochs 50 --n_epochs_decay 50
```

### Testing

To test the model on a new dataset, use the following command sample:

```
python test-new-eva.py --dataroot datasets/hazy2clear_0206/testA --name vit_512_100epoch_vgg --model test --no_dropout --results_dir ./result_new/ --num_test 300 --netG vit
```

This will test the model on the dataset located at `datasets/hazy2clear_0206/testA` and save the results in the specified `./result_new/` directory.

### Pretrained Model

Pretrained models are available in the `checkpoints/` folder. You can directly use the pretrained model for inference or further fine-tuning on your own data.