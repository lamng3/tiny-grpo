# tiny-grpo
A minimal, hackable implementation of Group Relative Policy Optimization (GRPO).

**Goal**: Provide a working implementation of GRPO for training a local `llama-3.2-3b` model using RL. Focus on understanding GRPO algorithm and running everything locally on a single RTX A6000 node on RunPod.

This project is inspired by and builds upon [open-thought/tiny-grpo](https://github.com/open-thought/tiny-grpo).

### Setup

1. Spin up a RunPod instance

```
choose 1 RTX A6000 ($0.49/hr) on Axolotl docker image.
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

5. Transfer file from local computer to RunPod instance

```
scp -i ~/.ssh/id_ed25519 -P <port> <local_file_path> root@<host_name>:<remote_destination_path>
```

### Training Results

coming soon ...

### References

- [DeepSeek-R1 tech report](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
