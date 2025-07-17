envpath="your_env_path_here"
conda create -p $envpath python=3.9 -y
conda init
conda activate $envpath
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
pip install numpy==1.23.0
pip install timm scipy matplotlib einops
