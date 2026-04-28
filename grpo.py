import json
import os
import random
import re
import sys
import time
import pickle

import deepspeed
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from math_verify import parse, verify, ExprExtractionConfig

from ref import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

Q_batch_size = 1
model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"
beta = 0.04
num_per_Q = 8
all_steps = 1000
max_prompt_length = 400   
save_steps = 100
compute_gen_logps = True
clip_param = 0.2
ref_server = "http://localhost:59875"

deepspeed_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size * num_per_Q,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': 
            return None
    except: 
        return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1]) # (num_per_Q, prompt_length + output_length)
    data['rewards'] = bytes_to_tensor(dd[2]) # (num_per_Q,)
    data['refs'] = bytes_to_tensor(dd[3]) # (num_per_Q, output_length)
    if len(dd) == 5: 
        data['gen_logps'] = bytes_to_tensor(dd[4]) # (num_per_Q, output_length)
    return data

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path, 
    torch_dtype=torch.bfloat16, 
    _attn_implementation="sdpa"
)
gen_model = model

with open("./QAs.pkl", "rb") as f:
    QAs = pickle.load(f)

generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True, # do_sample=False时永远选择概率最高的token，do_sample=True时根据概率分布采样，增加多样性
    temperature=0.9, 
    num_return_sequences=num_per_Q, # 控制一个query生成的response的数量
    pad_token_id=tokenizer.pad_token_id
)

system_prompt = ( # this prompt is exactly the same as the one in DeepSeek-R1 technical report
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. " 
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. " 
    "The reasoning process and answer are enclosed within <think>...</think> "
    "and <answer>...</answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)
def gen_answers(prompts):
    tip_text = []
    for prompt in prompts:
        tip_text.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ], 
                tokenize=False, 
                add_generation_prompt=True
            )
        )

    tip_inputs = tokenizer(
        tip_text, 
        return_tensors="pt", 
        padding=True, # pad_token_id的decode结果是"<|endoftext|>"
        padding_side="left", # 在左侧padding，使同一个batch不同长度的prompt结尾对齐
        add_special_tokens=False
    )

    prompt_length = tip_inputs["input_ids"].shape[-1]
    if prompt_length > max_prompt_length: 
        return []
    
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}
    with torch.inference_mode():
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    completion_ids = tip_completion_ids[:, prompt_length:]
    answers = [
        tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids
    ]
    return answers

def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) # 使用正则表达式在answer中查找所有数字
    if len(nums) == 0: return -1.0
    lastnum = nums[-1] # 用answer中最后一个数字和ground_truth做比较
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1

def reward_format(item, answer):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer><\|im_end\|>$" # 当前使用的Qwen2.5-7B-Instruct倾向于在</think>和<answer>间添加换行符，在</answer>后有表示生成结束的<|im_end|>。
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1

def gen_samples(inputs): # inputs的type是list，长度是Q_batch_size，default值为1。
    prompts = [input["Q"] for input in inputs]
    answers = gen_answers(prompts)
    if len(answers) == 0: 
        return None, None, None, None

    rewards = []
    for i, input in enumerate(inputs):
        for a in answers[i * num_per_Q : (i + 1) * num_per_Q]:
            rewards.append(reward_correct(input, a) + reward_format(input, a))
    prompts_text = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ], 
            tokenize=False, 
            add_generation_prompt=True
        ) for prompt in prompts
    ]
    prompt_inputs = tokenizer(
        prompts_text, 
        return_tensors="pt", 
        padding=True, 
        padding_side="left", 
        add_special_tokens=False
    )["input_ids"]
    output_ids = tokenizer(
        answers, 
        return_tensors="pt", 
        padding=True, 
        padding_side="right", 
        add_special_tokens=False
    )["input_ids"]
    return prompt_inputs, output_ids, torch.tensor(rewards, dtype=torch.float32), answers

def generate_mode(num=10, rank=0):
    if rank == 0: 
        print('enter generate mode')
    tic = time.time()
    for ii in range(num):
        inputs = random.sample(QAs, Q_batch_size) 
        # inputs的type是list，长度是Q_batch_size，每个元素是一个dict，包含'Q'和'A'两个key。
        # Q_batch_size的default值是1，即inputs中仅含有1个问答对。后续注释中均默认Q_batch_size=1。
        prompt_inputs, output_ids, rewards, answers = gen_samples(inputs) # 模型会对inputs中的1个Q，生成num_per_Q个response。
        # prompt_inputs: (1, prompt_length)
        # output_ids: (num_per_Q, output_length)
        if prompt_inputs is None: 
            continue
        if rank == 0: 
            print('rewards:', rewards)
            #if ii == 5: 
            #    print('answers:', answers[0])
        if (rewards.max() - rewards.min()).item() < 0.01:
            continue
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4) # (num_per_Q,)
        rep = output_ids.shape[0] // prompt_inputs.shape[0] # rep == num_per_Q
        prompt_length = prompt_inputs.shape[1]
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length) # (num_per_Q, prompt_length)
        merged_ids = torch.cat([Qrep, output_ids], dim=1) # (num_per_Q, prompt_length + output_length)
        data = [
            json.dumps({"plen": prompt_length}).encode(), 
            tensor_to_bytes(merged_ids), # (num_per_Q, prompt_length + output_length) 
            tensor_to_bytes(rewards) # (num_per_Q,)
        ]

        if compute_gen_logps:
            with torch.inference_mode():
                merged_ids = merged_ids.to(model.device) # (num_per_Q, prompt_length + output_length)
                gen_logps = get_per_token_logps(
                    model(merged_ids).logits[:, :-1, :], # (num_per_Q, prompt_length + output_length - 1, V) # exclude the logit of the last token
                    merged_ids[:, 1:] # (num_per_Q, prompt_length + output_length - 1) # exclude the first token and serves as the ground truth
                )
            data.append(tensor_to_bytes(gen_logps[:, prompt_length - 1:].cpu())) # (num_per_Q, output_length)

        data = make_bytes_list(data)
        requests.post(f"{ref_server}/upload", data=data)

    if rank == 0: 
        print('exit generate mode')
    print(f'rank {rank}: {time.time()-tic:.3f}s')

if 'genonly' in sys.argv:
    model.to('cuda')
    generate_mode(999999)
    sys.exit()

engine, optimizer, _, _ = deepspeed.initialize(
    config=deepspeed_config, 
    model=model, 
    model_parameters=model.parameters()
)
gen_model = engine

def get_per_token_logps(logits, target_ids):
    # logits: (B, L-1, V)
    # target_ids: (B, L-1)
    per_token_logps = []
    for logits_row, target_ids_row in zip(logits, target_ids):
        log_probs = logits_row.log_softmax(dim=-1) # (L-1, V)
        token_log_prob = torch.gather(log_probs, dim=1, index=target_ids_row.unsqueeze(1)).squeeze(1) # (L-1,)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps) # (B, L-1)

def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device) # (num_per_Q, prompt_length + output_length)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1) # (num_per_Q, 1) 同一个query采样的G个response中，每个response所有token的advantage相同。

    logits = engine(inputs).logits # (num_per_Q, prompt_length + output_length, V) = (B, L, V)
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
    
    per_token_logps = get_per_token_logps(logits, input_ids) # (B, L-1)
    per_token_logps = per_token_logps[:, prompt_length - 1:] # (num_per_Q, output_length)
    ref_per_token_logps = batch['refs'].to(per_token_logps.device) # (num_per_Q, output_length)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int() # .int()之前的部分是bool tensor，.int()将其转换为int类型的tensor，有效token位置为1，padding位置为0。

    if 'gen_logps' in batch:
        policy_ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device)) # (num_per_Q, output_length)
        clipped_policy_ratio = torch.clamp(policy_ratio, 1 - clip_param, 1 + clip_param)
        per_token_loss = torch.min(policy_ratio * advantages, clipped_policy_ratio * advantages) # (num_per_Q, output_length)
    else:
        assert compute_gen_logps is False
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = (
        (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
    ).mean()
    return loss

generate_mode(rank=torch.distributed.get_rank()) # 开始训练前，每个worker先进入一次generate_mode。

progress = range(1, all_steps + 1)
if torch.distributed.get_rank() == 0: 
    progress = tqdm(progress)
for step in progress: # range(1, all_steps + 1)
    batch = get_batch()
    while batch is None:
        generate_mode(rank=torch.distributed.get_rank())
        batch = get_batch()

    loss = GRPO_step(batch)
    engine.backward(loss)
    engine.step()

    if torch.distributed.get_rank() == 0:
        progress.set_description(f"Loss: {loss.item():.6f}")

    if step % save_steps == 0:
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            print('saving model')
            save_name = f"./step_{step}"
            state_dict = engine.module.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            engine.module.save_pretrained(save_name, state_dict=state_dict)
            tokenizer.save_pretrained(save_name)
        dist.barrier()