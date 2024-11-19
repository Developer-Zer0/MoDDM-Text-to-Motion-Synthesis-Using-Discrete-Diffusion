## Text-Motion Generator

Generating motions based on text descriptions. Analogy to text-to-image that generate new images from text. 

## Instructions to setup

```bash
conda create -n text2motion python=3.9
conda activate text2motion
# Clone repository recursively
git clone https://github.com/Developer-Zer0/MoDDM-Text-to-Motion-Synthesis-Using-Discrete-Diffusion.git --recurse-submodules
# Install Pytorch 1.10.0 (**CUDA 11.1**)
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# Install required pacakges
pip install -r requirements.txt
# Install DetUtil 
cd DetUtil
python setup.py develop
```

## Perform single sample inference
You need to setup FFMPEG for .mp4 generation. Follow instructions at https://www.ffmpeg.org/download.html
After installation, add path to ffmpeg.exe (inside bin folder) in .env (Rename .env.example).

API to run single sample inference using trained model on HumanML3D. Edit `sample_description.txt` to any text description of your choice. Inference **does not require GPU** and runs completely on CPU within 15 seconds. First run can take additional time to load CLIP. Run the following script and your human motion .mp4 will be stored in `generations/`.

```bash
python sample_generation.py
```

## Dataset
TODO: Add instructions for downloading HumanML3D and KIT-ML datasets
Default dataset will be the HumanML3D dataset. To use the KIT dataset add datamodule=kit-amass-rot.yaml

## Train Stage 1 Vector Quantized Variational AutoEncoder (VQ-VAE)


## Train Stage 2 Discrete Diffusion Model
and 
## Evaluate Model

## Training different baseline models
Values for MODEL-CONFIG can be:

```list
temos_model.yaml
guo_vae.yaml
tevet_diffusion.yaml
vq_vae.yaml
vq_diffusion.yaml
```
```bash
 python src/train.py --config-name=train model=MODEL-CONFIG model.do-evaluations=false trainer.devices=[1] trainer.max_epochs=500
 ```


## Rendered animations
Validation animations will be rendered after every 10 epochs starting from 0 and will be stored in logs/.
3 mp4 files will be created at each render step. Synthesis.mp4 will have the inference results,
single_step.mp4 (for diffusion) will have single step results, original.mp4 will have ground truth motions. 