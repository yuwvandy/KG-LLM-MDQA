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
from scripts.prepare_alpaca import generate_prompt

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
max_new_tokens = 100
top_k = 200
temperature = 0.8


def main(
    lora_path: Path = Path("out/lora/reason/7B/lora-finetuned.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama2/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama2/tokenizer.model"),
    quantize: Optional[str] = None,
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
    assert lora_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=[2], precision=precision)

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



from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

torch.set_float32_matmul_precision("high")
model, tokenizer = main()
@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json #['instruction': str, 'input': str]
    prompt = generate_prompt(data)

    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    output = generate(
            model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id
        )
    
    output = tokenizer.decode(output)
    output = output.split("### Response:")[1].strip()

    model.reset_cache()
    output.split('\n')

    return jsonify({'answer': output})
    



if __name__ == "__main__":
    app.run(port = 5000)
