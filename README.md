# tiny-grpo
A minimal, hackable implementation of Group Relative Policy Optimization (GRPO).

**Goal**: Implementation of GRPO for training a local `llama-3.2-3b` model using RL. Focus on understanding GRPO algorithm. Run everything locally with a single RTX A6000 node and Axolotl Docker image on RunPod.

This project is inspired by and builds upon [open-thought/tiny-grpo](https://github.com/open-thought/tiny-grpo).

### Updates

**[2025-07-19]** (in-progress) Supporting [GPG: Group Policy Gradient](https://arxiv.org/abs/2504.02546). Some configurations can be found in [verl/gpg](https://verl.readthedocs.io/en/latest/algo/gpg.html). The main task here is to update reward modeling, grouping advantages, remove KL divergence.

- Following [GPG/open-r1/src/open_r1/gpg_trainer.py](https://github.com/AMAP-ML/GPG/blob/main/open-r1/src/open_r1/gpg_trainer.py), we implemented `inverse_alpha`, avoiding divide by zero when `n_valid_samples = 0`.

- (in-progress) Resampling based on ratio of valid samples.

**[2025-07-16]** Upgrading `transformers 4.48.1 -> 4.53.2`. Starting from `transformers>=4.50`, the library modularized model support (see [huggingface/transformers (release v4.50.0)](https://github.com/huggingface/transformers/releases/tag/v4.50.0)). Switched to using `AutoModelForCausalLM`.

- To load LLaMA models, you must explicitly install the llama extra. Perform sanity check with `python -c "from transformers.models.llama import LlamaForCausalLM; print(LlamaForCausalLM)"`

- See [lamng3/tiny-grpo (issue #1)](https://github.com/lamng3/tiny-grpo/issues/1) for more details.

**[2025-07-15]** Supporting [DAPO](https://arxiv.org/abs/2503.14476), following this [huggingface/trl#3130 (comment)](https://github.com/huggingface/trl/issues/3130#issuecomment-2746947835).

- **Token-level Loss** is already implemented as `masked_mean`.     
- **Clip-Higher** is implemented following [huggingface/trl#3118 (comment)](https://github.com/huggingface/trl/pull/3118). DAPO recommends using `clip_eps_low = 0.2` and `clip_eps_high = 0.28`.
- **Dynamic Sampling (in progress)** is skipped in [huggingface/trl#3130 (comment)](https://github.com/huggingface/trl/issues/3130#issuecomment-2746947835) because of inefficiency. The configurations can be found in [verl/dapo](https://verl.readthedocs.io/en/latest/algo/dapo.html).
- **Overlong Filtering** is skipped (see the reason in [huggingface/trl#3130 (comment)](https://github.com/huggingface/trl/issues/3130#issuecomment-2746947835) and [verl/dapo](https://verl.readthedocs.io/en/latest/algo/dapo.html)).     
- **Soft Overlong Punishment** is implemented following [huggingface/trl#3130 (comment)](https://github.com/huggingface/trl/issues/3130#issuecomment-2746947835), with `L_cache = 256` for `L_max = 1024`, inspired from [verl/dapo](https://verl.readthedocs.io/en/latest/algo/dapo.html), where `L_cache` is `overlong_buffer`.

**[2025-07-12]** Supporting [Dr.GRPO](https://arxiv.org/abs/2503.20783), with modifications in calculating {`masked_mean` with constant generation max tokens (512 from [oat/oat/args.py](https://github.com/sail-sg/oat/blob/main/oat/args.py))} and {`group_advantage` without std bias}, following [understand-r1-zero/train_zero_math.py](https://github.com/sail-sg/understand-r1-zero/blob/main/train_zero_math.py#L288).

### Setup

1. Spin up a RunPod instance

```
choose 1 RTX A6000 ($0.49/hr) on Axolotl Docker image.
```

2. Create conda env

```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create --name grpo python=3.12 -y
source ~/.bashrc # or ~/.zshrc if you're using zsh
conda init
conda activate grpo
```

3. Install dependencies

```
cd tiny-grpo
pip install -r requirements.txt
pip install hf_transfer
pip install flash-attn --no-build-isolation
```

4. HuggingFace and WandB login

```
huggingface-cli login
wandb login
```

5. Play with the source in `train.py`

```
python train.py
```

6. Pushing code to GitHub

```
<!-- generate SSH key (if not yet) -->
ssh-keygen -t ed25519 -C "your_email@example.com"

<!-- add public key to github -->
cat ~/.ssh/id_ed25519.pub

<!-- change repo remote to SSH -->
git remote set-url origin git@github.com:<username>/<repo_name>.git

<!-- test connection -->
ssh -T git@github.com
```

7. Transfer file from local computer to RunPod instance

```
scp -i ~/.ssh/id_ed25519 -P <port> <local_file_path> root@<host_name>:<remote_destination_path>
```

### Training Results

The run stopped at 9K steps due to insufficient storage memory. To prevent this, consider doubling the storage capacity or offloading checkpoints to temporary storage.

<img src="results/returns.png" alt="Training Returns" width="800" height="600"/>

### References

- [GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning](https://arxiv.org/abs/2504.02546)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- [DeepSeek-R1 Tech Report](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
