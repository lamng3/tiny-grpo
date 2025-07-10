# tiny-grpo

### Setup

1. Spin up a Runpod instance

Choose 1 RTX A6000 on Axolotl docker image.

2. Create conda env

```
conda create --name grpo python=3.12 -y
source ~/.bashrc # or ~/.zshrc if you're using zsh
conda activate grpo
```

3. Install dependencies

```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

4. Play with the source in `train.py`

```
python train.py
```