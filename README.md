## Text-Motion Generator

![alt text](assets/Architecture.png)

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
API to run single sample inference using trained model on HumanML3D. Edit `sample_description.txt` to any text description of your choice. Inference **does not require GPU** and runs completely on CPU within 15 seconds. First run can take additional time to load CLIP.

1) You need to setup FFMPEG for .mp4 generation. Follow instructions at <a href='https://www.ffmpeg.org/download.html'>LINK</a>. After installation, add path to ffmpeg.exe (inside bin folder) in .env (Rename .env.example).

2) Download <a href='https://drive.google.com/file/d/1al0yAaOyUVx959W6hiyWyGVsDTq-rOJ3/view?usp=sharing'>autoencoder checkpoint</a> and <a href='https://drive.google.com/file/d/15igbR5bfv3E-fv1nKDd0pl1w7_41ZrOr/view?usp=sharing'>discrete diffusion checkpoint</a>. Store them under `checkpoints/` (Create if doesn't exist).

3) You will also need to download <a href='https://drive.google.com/file/d/1bzD_qzqwv4T5SKMKk7VvbzvXu61ee86H/view?usp=sharing'>SMPL_DATA</a> and <a href='https://drive.google.com/file/d/1wmmyIyBYegYQCh-MTbvrVZfnobgcMbTt/view?usp=sharing'>Deps</a> for the human skeleton transformations and animations. Extract them and store under `data/` (Create if doesn't exist) (`data/Deps`, `data/SMPL_DATA`).

4) Run the following script and your human motion .mp4 will be stored in `generations/`.

```bash
python sample_generation.py
```

## Dataset
To get both HumanML3D and KIT-ML dataset, follow instructions at https://github.com/EricGuo5513/HumanML3D. Once downloaded, store at location `data/` (Create if doesn't exist). For training and evaluations, you will also need SMPL_DATA and Deps from Step 3 of single sample inference.
Default dataset will be the HumanML3D dataset in all experiments. To use the KIT dataset add `datamodule=guo-kit-ml.yaml` as a parameter in command scripts.

## Train Stage 1 Vector Quantized Variational AutoEncoder (VQ-VAE)

Train VQ-VAE reconstruction model on HumanML3D (or KIT-ML). This stage 1 training is required for stage 2 diffusion. Run the following script.

```bash
 python src/train.py --config-name=train model=vq_vae.yaml model.do_evaluation=false trainer.devices=[1] trainer.max_epochs=500
 ```

Setting `model.do_evaluation=True` will run the evaluator after every epoch to store FID, R-Precision. However, evaluator is a pre-trained model by the work at https://github.com/EricGuo5513/TM2T. You will need to download the pre-trained models from <a href='https://drive.google.com/file/d/1OXy2FBhXrswT6zE4SBSPpVfQhxmI8Zzy/view'>LINK</a>. For HumanML3D evaluator, you need the `t2m/text_mot_match/model/finest.tar`. Store it at `checkpoints/t2m/text_mot_match/model/finest.tar`.

KIT-ML pre-trained models are from the above work as well and can be found at <a href='https://drive.google.com/file/d/1ied_KWvqXXsP2Gls-SvzjXIZtHHZ5zpi/view'>LINK</a>. For the KIT-ML evaluator, you need the `kit/text_mot_match/model/finest.tar`. Store is at `checkpoints/kit/text_mot_match/model/finest.tar`. Also include `eval_ckpt=checkpoints/kit/text_mot_match/model/finest.tar` as parameter in script.


## Train Stage 2 Discrete Diffusion Model

## Synthesized Motions