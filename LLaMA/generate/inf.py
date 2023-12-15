import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import json
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from core import Tokenizer, LLaMA
from core.lora import lora
from core.utils import lazy_load, llama_model_lookup
from core.utils import quantization

#***********hyperparameter***********
dataset  = 'cora'
task = 'none'
ft_method = 'lora'
size = '7B-chat'
devices = [1]
epoch = 25599

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
max_new_tokens = 256
top_k = 200
temperature = 1e-5

lora_path = Path(f"out/lora/{dataset}/{size}/iter-{epoch}-{task}-ckpt.pth")
# lora_path = Path(f"out/lora/{dataset}/{size}/llama2-{size}-lora-ft-{task}.pth")
pretrained_path = Path(f"checkpoints/lit-llama2/{size}/lit-llama2.pth")
tokenizer_path = Path("checkpoints/lit-llama2/tokenizer.model")
quantize = None

#************************************


def main(
    lora_path: Path,
    pretrained_path: Path,
    tokenizer_path: Path,
    quantize: Optional[str],
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        model_size: The model size to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    assert pretrained_path.is_file(), pretrained_path
    assert tokenizer_path.is_file(), tokenizer_path

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=devices, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    
    with lazy_load(pretrained_path) as pretrained_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), quantization(mode=quantize):
            model = LLaMA.from_name(name)
            model.load_state_dict(pretrained_checkpoint)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(tokenizer_path)

    return model, tokenizer 


def main_lora(
    lora_path: Path,
    pretrained_path: Path,
    tokenizer_path: Path,
    quantize: Optional[str],
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LoRA model.
    See `finetune_lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        lora_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    print(lora_path)
    assert lora_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=devices, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(lora_path) as lora_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(name)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            model.load_state_dict(lora_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(tokenizer_path)

    return model, tokenizer


# def generate_prompt_nei(example):
#     example['nei_texts'] = example['input'].split('[Contexts]: ')[1].split('[Text]: ')[0].split('\n\n')
#     example['input'] = '[Text]: ' + example['input'].split('[Text]: ')[1]
    
#     start = "Below is an instruction that describes a task, paired with an input that provides further context. "
#     middle = "Write a response that appropriately completes the request.\n\n"

#     instruction = "### Instruction:\nGiven the following Contents that is correlated to the [TEXT] to be completed, please complete [TEXT].\n\n"
#     context = "\n\n".join(["Content: {}".format(nei_text) for nei_text in example['nei_texts'][:3]])
#     last = '\n\n### Response:{}'.format(example['input'])

#     return start + middle + instruction + context + last

# def generate_prompt_none(example):
#     example['input'] = '[Text]: ' + example['input'].split('[Text]: ')[1]
    
#     start = "Below is an instruction that describes a task, paired with an input that provides further context. "
#     middle = "Write a response that appropriately completes the request.\n\n"

#     instruction = "### Instruction:\nGiven the following Contents that is correlated to the [TEXT] to be completed, please complete [TEXT].\n\n"
#     # context = "\n\n".join(["Content: {}".format(nei_text) for nei_text in example['nei_texts'][:3]])
#     last = '\n\n### Response:{}'.format(example['input'])

#     return start + middle + instruction + last


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an Instruction, optional Input and a Response field."""

    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    model, tokenizer = main_lora(lora_path, pretrained_path, tokenizer_path, quantize)

    test_data = torch.load(f'./data/{dataset}/test_{task}.pt')

    preds = {"task": "LaMP_4", "golds": []}
    for d in tqdm(test_data):
        sample = {"instruction": d['instruction'], \
                  "input": d['input']}
        
        # prompt = generate_prompt_none(sample)
        prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

        t0 = time.perf_counter()
        output = generate(
            model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id
        )
        t = time.perf_counter() - t0
        
        output = tokenizer.decode(output)
        output = output.split(prompt)[1]
        

        preds['golds'].append({'id': str(d['idx']), 'output': output.strip('###').strip('Response').strip(':[Text]')})
        # print(preds['golds'][-1])

        model.reset_cache()

    json.dump(preds, open(f'./out/lora/{dataset}/{size}/pred_llama2_{ft_method}_{task}_{max_new_tokens}_{epoch}.json', 'w'))