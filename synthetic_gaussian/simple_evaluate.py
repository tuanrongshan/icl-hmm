import argparse
import random
import numpy as np
import pandas as pd
import torch
import pandas as pd
import pickle
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MAX_SEQ_LEN = 100

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset", type=str, default="data/generations.pickle")
    parser.add_argument("--save_pickle", type=str, default="syn_results.pickle")
    parser.add_argument("--save_csv", type=str, default="syn_results.csv")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--selected_std", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11111)
    parser.add_argument("--num_seq", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=8)
    return parser.parse_args()

def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

args = parse_arguments()
set_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16
).to(device)
model.eval()

with open(args.dataset, 'rb') as file:
    object_file = pickle.load(file)
num_states, lambda2s, Us, Sigmas, U_invs, As, A_entropys, observations, hidden_states, means_list, stds_list, pi_0s = object_file


def inference_batch(batch_obs):
    prompts = []
    for obs in batch_obs:
        obs_str = ", ".join([f"{x}" for x in obs])
        prompt = (
            f"Predict the next number in the sequence below.\n"
            f"Output JSON only in the format: {{\"next_value\": number}}\n"
            f"Data: [{obs_str}]"
        )
        prompts.append(prompt)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = []
    for response in responses:
        try:
            matches = re.findall(r'\{.*?\}', response)
            res = None
            for m in matches:
                try:
                    val = json.loads(m).get("next_value", None)
                    if isinstance(val, (int, float)):
                        res = float(val)
                        break
                except:
                    pass
            results.append(res)
        except:
            results.append(None)
        
    return results


data_size = len(num_states) // 7
entropys = [args.selected_std] * data_size
predictions = []
pdfs = []

data_l = data_size * args.selected_std
data_r = data_size * (args.selected_std + 1)
batch_size = args.batch_size
for i in tqdm(range(data_l, data_r, batch_size)):
    batch_obs = [obs[:(MAX_SEQ_LEN-1)] for obs in observations[i:i+batch_size]]
    batch_hidden = hidden_states[i:i+batch_size]
    batch_means = means_list[i:i+batch_size]
    batch_stds = stds_list[i:i+batch_size]

    batch_predict = inference_batch(batch_obs)
    predictions.extend(batch_predict)

    def gaussian_pdf(x, mean, std) -> float:
        return float(min(1., (1.0 / (np.sqrt(2 * np.pi) * std + 1e-12)) * np.exp(-0.5 * ((x - mean) / (std + 1e-12)) ** 2)))
    
    for predict, hidden_state, means, stds in zip(batch_predict, batch_hidden, batch_means, batch_stds):
        if predict is not None:
            state_label = int(hidden_state[MAX_SEQ_LEN-1])
            pdfs.append(gaussian_pdf(predict, means[state_label], stds[state_label]))
        else:
            pdfs.append(None)

data = pd.DataFrame({
    "num_states": num_states[data_l:data_r],
    "lambda2s": lambda2s[data_l:data_r],
    "Us": Us[data_l:data_r],
    "Sigmas": Sigmas[data_l:data_r],
    "U_invs": U_invs[data_l:data_r],
    "As": As[data_l:data_r],
    "A_entropys": A_entropys[data_l:data_r],
    "observations": observations[data_l:data_r],
    "hidden_states": hidden_states[data_l:data_r],
    "means_list": means_list[data_l:data_r],
    "stds_list": stds_list[data_l:data_r],
    "pi_0s": pi_0s[data_l:data_r],
    "entropys": entropys,
    "predictions": predictions,
    "pdfs": pdfs
})
csv_name = "eval_llm_" + str(args.selected_std) + ".csv"
data.to_csv("data/" + csv_name, index=False)