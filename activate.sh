env_name="tvr"
conda activate "your_env_path_here"  # Replace with your actual environment path
conda env config vars set CONDA_SHORT_NAME=$env_name
export PS1='\[\e[36m\]($CONDA_SHORT_NAME)\[\e[0m\] \w\n >>> '
export HF_ENDPOINT=https://hf-mirror.com