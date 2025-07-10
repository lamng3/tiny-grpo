# tiny-grpo

### Setup

1. Spin up a Runpod instance

```
choose 1 RTX A6000 on Axolotl docker image.
```

2. Create conda env

```
conda create --name grpo python=3.12 -y
source ~/.bashrc # or ~/.zshrc if you're using zsh
conda activate grpo
```

3. Install dependencies

```
pip install -r requirements.txt
pip install hf_transfer
pip install flash-attn --no-build-isolation
```

4. Play with the source in `train.py`

```
python train.py
```

5. Transfer file from local computer to Runpod instance

```
scp -i ~/.ssh/id_ed25519 -P <port> <local_file_path> root@<host_name>:<remote_destination_path>
```