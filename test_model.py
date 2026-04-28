from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import re
from math_verify import parse, verify, ExprExtractionConfig
import pickle
import time

model_name_or_path = ""
num_per_Q = 1
max_prompt_length = 512

with open("./QAs.pkl", "rb") as f:
    QAs = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path, 
    torch_dtype=torch.bfloat16, 
    _attn_implementation="sdpa"
)
gen_model = model.to('cuda')

generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=False, # do_sample=False时永远选择概率最高的token，do_sample=True时根据概率分布采样
    num_return_sequences=num_per_Q, # 一个query生成的response的数量
    pad_token_id=tokenizer.pad_token_id
)

def reward_format(item, answer):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer><\|im_end\|>$"
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1

def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer) # 使用正则表达式在answer中查找所有数字
    if len(nums) == 0: return -1.0
    lastnum = nums[-1] # 用answer中最后一个数字和ground_truth做比较
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1

system_prompt = (
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
        padding=True,
        padding_side="left",
        add_special_tokens=False
    )

    prompt_length = tip_inputs["input_ids"].shape[-1]
    if prompt_length > max_prompt_length: 
        return [], []

    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}
    with torch.inference_mode():
        tip_completion_ids = gen_model.generate(
            **tip_inputs,
            generation_config=generation_config
        )
    completion_ids = tip_completion_ids[:, prompt_length:]

    answers = []
    lengths = []
    for x in completion_ids:
        valid_tokens = x[x != tokenizer.pad_token_id] # pad_token_id的decode结果是"<|endoftext|>"。
        lengths.append(len(valid_tokens))
        answers.append(tokenizer.decode(valid_tokens))

    return answers, lengths

from tqdm import tqdm

if __name__ == "__main__":
    num_test_q = 500
    torch.manual_seed(42)
    sample_indices = torch.randperm(len(QAs))[:num_test_q]
    correct_record = torch.zeros(len(QAs))
    format_record = torch.zeros(len(QAs))
    average_length = 0

    for i, idx in enumerate(tqdm(sample_indices, desc="Evaluating", ncols=100)):
        t = time.time()
        item = QAs[idx]
        prompt = item["Q"]
        answers, lengths = gen_answers([prompt])
        time_cost = time.time() - t

        for j, answer in enumerate(answers):
            correct_reward = reward_correct(item, answer)
            format_reward = reward_format(item, answer)
            if format_reward > 0:
                format_record[idx] = 1
            if correct_reward > 0:
                correct_record[idx] = 1
                break

        average_length += sum(lengths)

        tqdm.write(
            f"Question {idx}, correct reward: {correct_reward}, "
            f"format reward: {format_reward}, time: {time_cost:.2f}s"
        )

    print(f"Pass@{num_per_Q}: {torch.sum(correct_record).item() / num_test_q:.4f}")
    print(f"Format@{num_per_Q}: {torch.sum(format_record).item() / num_test_q:.4f}")
    print(f"Average length: {average_length / (num_test_q * num_per_Q):.2f}")