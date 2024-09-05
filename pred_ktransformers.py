import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import platform
import yaml

from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.util.utils import prefill_and_generate
from ktransformers.server.config.config import Config

# Data and methods for ktransformers -----------------------------------------
with open("config_ktransformers.yaml", "r") as file:
    config_ktr = yaml.safe_load(file)

longbench_script = config_ktr['longbench_script']
internlm_gguf_path = config_ktr['internlm_gguf_path']
ktransformer_home = config_ktr['ktransformer_home']
ktransformer_rules_dir = config_ktr['ktransformer_rules_dir']

default_optimize_rules = {
    # "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    # "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    # "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml",
}

def load_model_and_tokenizer(model_path, device, gguf_path):
    assert "internlm" in model_name

    # adopt from ktransformers/localchat.py
    torch.set_grad_enabled(False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    arch = config.architectures[0]
    print(model_path, arch)
    assert arch == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
    torch.set_default_dtype(torch.float16)

    with torch.device("meta"):
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config)
    
    assert arch in default_optimize_rules
    optimize_rule_path = default_optimize_rules[arch]

    assert gguf_path is not None
    optimize_and_load_gguf(model, optimize_rule_path, gguf_path, config)
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.eos_token
    model.eval()
    logging.basicConfig(level=logging.INFO)
    
    return model, tokenizer
# -----------------------------------------------------------------------------

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "internlm2_5-7b-chat-1m"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    assert "internlm" in model_name
    prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    assert "internlm" in model_name
    response = response.split("<eoa>")[0]
    return response

def get_pred(data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    # device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name], 
        model_name, 
        gguf_path=internlm_gguf_path)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if "chatglm3" in model_name:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: 
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        # context_length = input.input_ids.shape[-1]
        
        input_tensor = input["input_ids"]
        assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_gen, \
            "please change max_seq_len in  ~/.ktransformers/config.yaml"
        generated_tokens = prefill_and_generate(
            model=model, 
            tokenizer=tokenizer, 
            inputs=input_tensor, 
            max_new_tokens=max_gen,
            use_cuda_graph=True,
            mode="long_context")
        
        # output = model.generate(
        #     **input,
        #     max_new_tokens=max_gen,
        #     num_beams=1,
        #     do_sample=False,
        #     temperature=1.0,
        # )[0]
        pred = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    # dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = "internlm2_5-7b-chat-1m"
    max_length = model2maxlen[model_name]

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset(longbench_script, f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset(longbench_script, dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        # max_gen = dataset2maxlen[dataset]
        max_gen = 256 # max generated, not max_seq_len
        data_all = [data_sample for data_sample in data]
        
        get_pred(data=data_all, 
                 max_length=max_length, 
                 max_gen=max_gen, 
                 prompt_format=prompt_format, 
                 dataset=dataset, 
                 device=device, 
                 model_name=model_name,
                 model2path=model2path,
                 out_path=out_path)
