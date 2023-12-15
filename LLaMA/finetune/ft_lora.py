"""
Instruction-tuning with LoRA.
"""
import sys
from pathlib import Path
import os
import time
import lightning as L
# from prepare_col import generate_prompt
import numpy as np
import torch
import wandb

wandb.login()

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from core.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from core.model import LLaMA, LLaMAConfig, LLaMA2Config
from core.tokenizer import Tokenizer

#***********hyperparameter***********
torch.set_float32_matmul_precision("high")
eval_interval, save_interval, log_interval = 100, 100, 1
warmup_iters, eval_iters = 100, 100

batch_size, micro_batch_size = 1024, 8
gradient_accumulation_iters = batch_size // micro_batch_size
max_iters = (gradient_accumulation_iters * save_interval)*20

learning_rate = 3e-4
weight_decay = 0.0
max_new_tokens = 100

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
device_type = "cuda"
device_num = 8
precision = "bf16-true"


#specific for reason dataset
dataset = 'reason'
size = '7B'
data_dir = f"data/{dataset}"
max_seq_length = 1024
pretrained_path = f"checkpoints/lit-llama2/{size}/lit-llama2.pth"
tokenizer_path = "checkpoints/lit-llama2/tokenizer.model"
out_dir = f"out/lora/{dataset}"


#wandb
run = wandb.init(project = f'lora_ft_llama2-{size}_{dataset}')
columns = ['Input', 'True Output', 'Predicted Output']
#***********hyperparameter***********


def main() -> None:
    fabric = L.Fabric(accelerator = device_type, devices = device_num, precision = precision)
    fabric.launch()
    fabric.seed_everything(1028 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir = data_dir)
    print(len(train_data), len(val_data))

    config = LLaMA2Config.from_name("7B")    # config = LLaMAConfig.from_name("7B")
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with fabric.init_module(), lora(r = lora_r, alpha = lora_alpha, dropout = lora_dropout, enabled = True):
        model = LLaMA(config)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False)
    
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, f"./{size}/lora-ft.pth"), checkpoint)


def train(fabric: L.Fabric, model: torch.nn.Module, optimizer: torch.optim.Optimizer, \
          train_data: np.ndarray, val_data: np.ndarray) -> None:
    """The training loop.
    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step = 0

    for iter_num in range(max_iters):

        if step <= warmup_iters:
            # linear warmup
            lr = learning_rate * step / warmup_iters

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step += 1
                
            if step % eval_interval == 0:
                val_loss = validate(fabric, model, val_data, step)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step % save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                checkpoint = lora_state_dict(model)
                fabric.save(os.path.join(out_dir, f"./{size}/iter-{iter_num}-ckpt.pth"), checkpoint)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
            wandb.log({'loss': loss.item()})
            # val_loss = validate(fabric, model, val_data, step)
            # wandb.log({'val loss': val_loss})


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an Instruction, optional Input and a Response field."""

    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
    )


def generate_response(model, sample):
    tokenizer = Tokenizer(tokenizer_path)
    
    prompt = generate_prompt(sample)

    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx = encoded,
        max_seq_length = max_seq_length,
        max_new_tokens = max_new_tokens,
    )

    output = tokenizer.decode(output)

    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, step: int) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # if task == "none":
    #     sample = {"instruction": "Complete the [Text]", \
    #               "input": "[Text]: i bought my 99 millenia with the premium package and all weather package in 2000 so far i have about 13000 miles on it overall it is a nice car the acceleration is smooth and adequate the car shifts nicely without hesitation or clunking the road noise is fairly low the car handles well and is very tight turns the braking is excellent in both dry and wet weather the michele tires grip on most surfaces very well except snow or ice however they are prone to frequent flats living in nyc i judge a cars quality by how well it holds togethor after hitting potholes and frequent bumpy roads due to road construction the car has held togethor pretty well the car hardly rattles on uneven pavement and the suspension still feels tight like the day i drove it off the showroom floor as far as maintenance all i",\
    #               "output": "have done is change the oil every 3000 miles or three months the car has been trouble free except two flats the interior of the car is very comfortable the power seats allow ample adjustments to fit your personal preferences the steering wheel tilts upward when you get out of the car the moonroof is not too noisy when it is open the heated seats work very well and heat up the leather fairly quickly the wipers can be adjusted with easy and they have no problem keeping up with light or heavy rain the stereo system is clear and loud it includes ive speakers as well as a cd and cassette there is no antennae to get broken when going through the car wash because it is built into the window the washe fluid bottle holds a gallon some of the drawbacks are that it could use fold down rear seats to allow the carrying of more cargo the tires could be harder to prevent such easy flats it could use a little more power it could use some more glove box space as well as tether ancors for baby seats while the car is fine the addition of a tether hookup for child safety seats would be novel idea it could use a bigger pass through on the back seat or a fold down back seat trunk space is limited the rear view mirror seems a bit low i am 510 and find it is always in my line of vision it could use body side moldings the car dings too easily also the weatherstripping on the moonroof could be better somtimes it rattlesi do wish the drivers seat moved a little further back as far as reliability i have had not really had any problems so far the only thing that broke is the horn i have to make an appointment to have it fixed under waranty"}
    # elif task in ['topo', 'topo*']:
    #     sample = {"instruction": "Refer to the relevant [Contexts] and Complete the [Text]", \
    #               "input": "[Contexts]: i came across the 2007 toyota corolla while looking around for a new car with my father we ended up at the toyota dealership where he had bought his last car which was a 94 corolla the 94 currently has 12500 miles on it now and is still running great but my brother a new driver is getting the 94 corolla passed down to him i expected to see many similarities between the 1994 and 2007 toyota corollas and i was not disappointed after taking a test drive of the car i found that the car had excellent handling and acceleration from the compact car that it was it handled sharp turns with ease and stuck to the ground while driving however i noticed that the corollas small engine struggled on big inclines which means the foot goes down harder on the gas which means it takes more gas to operate in hilly areas especially with a load of people being the compact car that it is the toyota corolla offers little room for and backseat passengers meaning that long trips would be uncomfortable for anyone over 5 feet tall even though it is a safe ride with side curtain airbags i would not recommend making this the family car a major plus in todays market is the miles per gallon of gasoline that cars can have today the 2007 corolla was tested and found to have a 38 highway miles per gallon needless to say my father bought the car and verified that the mpg was true as he was able to get 36 mpg he said that the easy acceleration of the car made it easy to keep up with traffic and use as little gas as possible a major plus with gas more than 3 per gallon i asked my dad why he bought his 2007 toyota corolla and he told me that his past experience with his old corolla helped him decide to buy a new one he as many others have found that toyota makes a very safe and reliable car that has great fuel economy which ends up saving the consumer a lot of money these reasons prove why the toyota corolla is the best selling car in history\n\n[Text]: i bought this mp3 player as a replacement for my old sony mp3 player see my review for the nw a1000 and at first was a bit disappointed that i only got a 1gb player for the same cost as my old one but having used it for just over two weeks im convinced that i made the right choice using the player is really quite simple once you get used to the controls but it does take a little bit of time to adjust to using the shuttle switch and mode key to navigate back and forth through the menus as well as standard navigation the player also uses its colour screen to full advantage and allows you to search by album artwork a lot of people complain about sonys sonicstage software that must be used to transfer the music to the player but that said i have never",\
    #               "output": "had any problems with using it and it seemed pretty intuitive to use within about 10 minutes i had installed the software and transferred several albums it supports sonys atrac 3 plus file format as well as standard mp3 but with the atrac format a song at 64 kbps has the same sound quality as an mp3 at 128 kbps so you can fit double the amount of music when you use it the player also features a noise cancelling feature which is great at reducing the volume of outside noise the earphones supplied with the unit may look like some medieval torture device but not only are they really comfortable but the sound quality is amazing too if you buy these earphones separately they cost 60 115 so to have them as standard is brilliant the battery life is great too since ive had the player i only charged it completely on the first day and the only other time its been connected is for about 10 minutes to transfer some more music to the unit i use the player for 34 hours daily so the battery life has been close to the stated 50 hours per charge the player also features a rapid charge feature so 3 minutes charge provides 3 hours of playback why did i buy it looks sony reliability noise cancelling feature colour organic led screen what do i love about it pretty much everything what do i hate the fact that i have to charge it from my computer as no ac adapter is supplied with it"}

    sample = {"instruction": "What evidence do we need to answer the question given the current evidence?", \
              "input:": "Which magazine was started first Arthur's Magazine or First for Women?\nArthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.",\
              "output": "First for Women is a woman's magazine published by Bauer Media Group in the USA."}

    # produce an example:
    output = generate_response(model, sample)
    output = output.split("### Response:")[1].strip()
    output = wandb.Table(data = [[sample['input'], sample['output'], output]], columns = columns)
    wandb.log({'examples': output})

    model.train()
    return out.item()


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous() # B x T x |W|
    targets = targets[..., 1:].contiguous() # B x T
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    
    return loss
    

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)

        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))

    return x, y


def load_datasets(data_dir: str):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "val.pt"))

    # print(train_data[0])

    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    from jsonargparse.cli import CLI

    CLI(main)