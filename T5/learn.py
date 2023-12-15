from tqdm import tqdm
import torch
import numpy as np


def train(epoch, tokenizer, model, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function
    """

    model.train()
    losses = []
    for _, data in enumerate(tqdm(loader, desc = f"Epoch {epoch}"), 0):
        y = data["target_ids"].cuda()
        # .to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].cuda()
        # .to(device, dtype=torch.long)
        mask = data["source_mask"].cuda()
        # .to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0].sum()

        

        # if _ % 10 == 0:
        #     print(str(epoch), str(_), str(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())


    return np.mean(losses)


def eval(tokenizer, model, loader):
    model.eval()
    predictions = []
    actuals = []
    questions = []

    with torch.no_grad():
        for _, data in enumerate(tqdm(loader, desc = f"Eval"), 0):
            y = data['target_ids'].cuda()
            # .to(device, dtype = torch.long)
            ids = data['source_ids'].cuda()
            # .to(device, dtype = torch.long)
            mask = data['source_mask'].cuda()
            # .to(device, dtype = torch.long)

            generated_ids = model.module.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=512, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]

            predictions.extend(preds)
            actuals.extend(target)
            questions.extend(data['source_text'])

            
    return predictions, actuals, questions