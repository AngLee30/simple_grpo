import json
import os
import io
import torch
import threading
import queue
from bottle import request, Bottle
from transformers import AutoModelForCausalLM

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)

def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()

def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        len = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(len))
    return blist

if __name__ == '__main__':   
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"

    ref_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        torch_dtype=torch.bfloat16, 
        _attn_implementation="sdpa"
    ).to('cuda') # ref_model is an instance of PreTrainedModel, which is a subclass of nn.Module
    ref_model.eval()
    ref_model.requires_grad_(False)

    def get_per_token_logps(input_ids): # input_ids shape: (B, L)
        logits = ref_model(input_ids).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            # logits_row shape: (L-1, V), input_ids_row shape: (L-1,)
            log_probs = logits_row.log_softmax(dim=-1) # (L-1, V)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1) # (L-1,) each element at position t is the log prob of the token at position t in input_ids_row given the previous tokens 
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps) # (B, L-1)

    raw_queue = queue.LifoQueue()
    result_queue = queue.LifoQueue()

    app = Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
        if len(dd) not in (3, 4): return b'tensor'
        data = {'base': json.loads(dd[0])} 
        data['inputs'] = bytes_to_tensor(dd[1]) # (num_per_Q, prompt_length + output_length)
        data['rewards'] = bytes_to_tensor(dd[2]) # (num_per_Q,)
        if len(dd) == 4: 
            data['gen_logps'] = bytes_to_tensor(dd[3]) # (num_per_Q, output_length)
        raw_queue.put(data)
        print(
            'receive', 
            "input shape:", data['inputs'].shape, 
            "rewards:", data['rewards'], 
            "gen_logps shape:", data['gen_logps'].shape if 'gen_logps' in data else ''
        )
        return b'tensor'

    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty(): 
            return b'empty'
        return result_queue.get() # remove and return an item from the queue
    
    def run_server(): Bottle.run(app, host='0.0.0.0', port=59875, server='tornado')
    threading.Thread(target=run_server, daemon=False).start()

    while True:
        d = raw_queue.get()
        prompt_length = d['base']['plen']
        with torch.inference_mode():
            per_token_logps = get_per_token_logps(d['inputs'].to(ref_model.device))
        per_token_logps = per_token_logps[:, prompt_length - 1:] # (num_per_Q, output_length)
        data = [
            json.dumps(d['base']).encode(), 
            tensor_to_bytes(d['inputs']), 
            tensor_to_bytes(d['rewards']), 
            tensor_to_bytes(per_token_logps)
        ]
        if 'gen_logps' in d: data.append(tensor_to_bytes(d['gen_logps']))
        data = make_bytes_list(data)
        result_queue.put(data)