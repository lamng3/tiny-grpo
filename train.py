from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM
)


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True, #bfloat16 data type
    device_map=None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    """load model with flash attention implementation"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token # add ending sentence token
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""



