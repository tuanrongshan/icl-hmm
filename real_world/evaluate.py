import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

RESULTS_PATH = 'results/'
os.makedirs(RESULTS_PATH, exist_ok=True)

def evaluate_llm(model, tokenizer, names, trajectories, eval_locations, candidates):

    # evaluate
    results = {}
    with torch.no_grad():
        for name, trajs in tqdm(zip(names, trajectories), total=len(trajectories)):

            results[name] = {
                'accuracy': [],
                'norm_likelihood': [],
            }

            for traj in trajs:

                # prepare batch
                traj = torch.from_numpy(np.array([traj])).long().to(model.device)
                eval_loc = np.arange(start=eval_locations[0], stop=traj.shape[1], step=eval_locations[1])

                # gather prob
                output = model(traj, return_dict=True)
                logits = output.logits[:, eval_loc - 1]
                all_logprob = F.log_softmax(logits, dim=-1)[:, :, np.array(candidates)]
                all_prob = torch.exp(all_logprob)
                all_prob = all_prob / all_prob.sum(-1, keepdim=True)

                # compute accuracy
                predicted_emission = torch.argmax(all_prob, dim=-1).cpu().numpy()
                traj = traj.cpu().numpy()
                results[name]['accuracy'].append((np.array(candidates)[predicted_emission][0] == traj[0, eval_loc]).tolist())

                # compute norm likelihood: exp(1/N (\sum_{i=1}^N log p_i))
                index = traj[:, eval_loc].T == np.array([candidates])
                all_prob = all_prob.float().cpu().numpy()[0]
                all_prob = all_prob[index]
                all_logprob = np.log(all_prob)
                results[name]['norm_likelihood'].append(np.exp(all_logprob.sum() / len(all_logprob)).astype(float))

    return results


def main():

    # define models
    models = ['meta-llama/Llama-3.2-1B', 'meta-llama/Llama-3.2-3B', 'meta-llama/Llama-3.1-8B', 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-7B']

    # load json
    with open('ibl_data/ibl_data_by_animal_dict.json', 'r', encoding='utf-8') as f:
        ibl_data = json.load(f)
    with open('rew_learn_data/tab_dataset.json', 'r', encoding='utf-8') as f:
        rew_learn_data = json.load(f)

    # process ibl dataset
    ibl_all_stim, ibl_all_previous_choice, ibl_all_rewarded, ibl_all_mouse_name = [], [], [], []
    for k, v in ibl_data.items():

        # gather the current mouse
        curr_stim = []
        curr_previous_choice = []
        curr_rewarded = []
        curr_session = []
        for d in v:
            curr_stim.append(d['stim'])
            curr_previous_choice.append(d['previous_choice'])
            curr_rewarded.append(d['rewarded'])
            curr_session.append(d['session'])
        curr_stim = np.array(curr_stim)
        curr_previous_choice = np.array(curr_previous_choice)
        curr_rewarded = np.array(curr_rewarded)
        curr_session = np.array(curr_session)

        # sort the trajectoies based on session date
        session_idx = np.argsort(curr_session)
        curr_stim = curr_stim[session_idx]
        curr_previous_choice = curr_previous_choice[session_idx]
        curr_rewarded = curr_rewarded[session_idx]

        curr_stim = np.expand_dims(curr_stim.flatten(), axis=0)
        curr_previous_choice = np.expand_dims(curr_previous_choice.flatten(), axis=0)
        curr_rewarded = np.expand_dims(curr_rewarded.flatten(), axis=0)

        # add to the dataset
        ibl_all_stim.append(curr_stim)
        ibl_all_previous_choice.append(curr_previous_choice)
        ibl_all_rewarded.append(curr_rewarded)
        ibl_all_mouse_name.append(k)

    # process rew learn dataset
    rew_learn_all_sides, rew_learn_all_rewards, rew_learn_all_mouse_name = [], [], []
    for i in range(len(rew_learn_data)):
        rew_learn_all_sides.append(np.array([list(rew_learn_data[i]['sides'])]))
        rew_learn_all_rewards.append(np.array([rew_learn_data[i]['rewards']]))
        rew_learn_all_mouse_name.append(rew_learn_data[i]['ratname'])

    # filter out v actions
    for i in range(len(rew_learn_all_sides)):
        mask = rew_learn_all_sides[i] != 'v'
        rew_learn_all_sides[i] = np.array([rew_learn_all_sides[i][mask]])
        rew_learn_all_rewards[i] = np.array([rew_learn_all_rewards[i][mask]])

    # filter to the first 16384
    for i in range(len(rew_learn_all_sides)):
        if rew_learn_all_sides[i].shape[1] > 16384:
            rew_learn_all_sides[i] = rew_learn_all_sides[i][:, :16384]
            rew_learn_all_rewards[i] = rew_learn_all_rewards[i][:, :16384]

    # ibl mapping
    ibl_stim_mapping = {
        -1. : ' A',
        -0.25 : ' B',
        -0.125 : ' C',
        -0.0625 : ' D',
        0. : ' E',
        0.0625 : ' F',
        0.125 : ' G',
        0.25 : ' H',
        1. : ' I',
    }
    ibl_previous_choice_mapping = {
        0 : ' J',
        1 : ' K',
    }
    ibl_rewarded_mapping = {
        -1 : ' L',
        1  : ' M',
    }

    # rew learn mapping
    rew_learn_side_mapping = {
        'r' : ' R',
        'l' : ' L',
    }
    rew_learn_reward_mapping = {
        0 : ' A',
        1 : ' B',
    }

    # evaluate datasets
    ibl_choice_only_results, ibl_stim_choice_results, ibl_choice_reward_results, ibl_stim_choice_reward_results = {}, {}, {}, {}
    rew_learn_side_only_results, rew_learn_side_reward_results = {}, {}
    for model_name in models:

        print(f'Evaluating {model_name}...')

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # compute token id
        if 'Qwen' in model_name:
            token_idx = 0
        else:
            token_idx = 1
    
        ibl_curr_stim_mapping = {k: tokenizer(v)['input_ids'][token_idx] for k, v in ibl_stim_mapping.items()}
        ibl_curr_previous_choice_mapping = {k: tokenizer(v)['input_ids'][token_idx] for k, v in ibl_previous_choice_mapping.items()}
        ibl_curr_rewarded_mapping = {k: tokenizer(v)['input_ids'][token_idx] for k, v in ibl_rewarded_mapping.items()}
        ibl_stim_mapper = np.vectorize(ibl_curr_stim_mapping.get)
        ibl_previous_choice_mapper = np.vectorize(ibl_curr_previous_choice_mapping.get)
        ibl_rewarded_mapper = np.vectorize(ibl_curr_rewarded_mapping.get)

        # map ibl dataset to token id
        ibl_all_stim, ibl_all_previous_choice, ibl_all_rewarded = [], [], []
        for s, p, r in zip(ibl_all_stim, ibl_all_previous_choice, ibl_all_rewarded):
            ibl_all_stim.append(ibl_stim_mapper(s))
            ibl_all_previous_choice.append(ibl_previous_choice_mapper(p))
            ibl_all_rewarded.append(ibl_rewarded_mapper(r))

        # prepare ibl dataset
        ibl_choice_only = ibl_all_previous_choice
        ibl_stim_choice, ibl_choice_reward, ibl_stim_choice_reward = [], [], []
        for s, p, r in zip(ibl_all_stim, ibl_all_previous_choice, ibl_all_rewarded):
            ibl_curr_stim_choice, ibl_curr_choice_reward, ibl_curr_stim_choice_reward = [], [], []
            for i in range(s.shape[1]):
                ibl_curr_stim_choice.extend([s[:, i], p[:, i]])
                ibl_curr_choice_reward.extend([p[:, i], r[:, i]])
                ibl_curr_stim_choice_reward.extend([s[:, i], p[:, i], r[:, i]])
            ibl_stim_choice = np.array(ibl_stim_choice).T
            ibl_choice_reward = np.array(ibl_choice_reward).T
            ibl_stim_choice_reward = np.array(ibl_stim_choice_reward).T
            ibl_stim_choice.append(ibl_curr_stim_choice)
            ibl_choice_reward.append(ibl_curr_choice_reward)
            ibl_stim_choice_reward.append(ibl_curr_stim_choice_reward)

        # ibl eval locations (start index, next index)
        ibl_choice_only_eval_loc = (0, 1)
        ibl_stim_choice_eval_loc = (1, 2)
        ibl_choice_reward_eval_loc = (0, 2)
        ibl_stim_choice_reward_eval_loc = (1, 3)

        rew_learn_curr_side_mapping = {k: tokenizer(v)['input_ids'][token_idx] for k, v in rew_learn_side_mapping.items()}
        rew_learn_curr_reward_mapping = {k: tokenizer(v)['input_ids'][token_idx] for k, v in rew_learn_reward_mapping.items()}
        rew_learn_side_mapper = np.vectorize(rew_learn_curr_side_mapping.get)
        rew_learn_reward_mapper = np.vectorize(rew_learn_curr_reward_mapping.get)
        
        # map dataset to token id
        rew_learn_all_sides, rew_learn_all_rewards = [], []
        for s, r in zip(rew_learn_all_sides, rew_learn_all_rewards):
            rew_learn_all_sides.append(rew_learn_side_mapper(s))
            rew_learn_all_rewards.append(rew_learn_reward_mapper(r))

        # prepare dataset
        rew_learn_side_only = rew_learn_all_sides
        rew_learn_side_reward = []
        for s, r in zip(rew_learn_all_sides, rew_learn_all_rewards):
            curr_side_reward = []
            for i in range(s.shape[1]):
                curr_side_reward.extend([s[:, i], r[:, i]])
            curr_side_reward = np.array(curr_side_reward).T
            rew_learn_side_reward.append(curr_side_reward)

        # eval locations (start index, next index)
        rew_learn_side_only_eval_loc = (0, 1)
        rew_learn_side_reward_eval_loc = (0, 2)

        # evaluate
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
        ibl_choice_only_results[model_name] = evaluate_llm(model, tokenizer, ibl_all_mouse_name, ibl_choice_only, ibl_choice_only_eval_loc, list(ibl_curr_previous_choice_mapping.values()))
        ibl_stim_choice_results[model_name] = evaluate_llm(model, tokenizer, ibl_all_mouse_name, ibl_stim_choice, ibl_stim_choice_eval_loc, list(ibl_curr_previous_choice_mapping.values()))
        ibl_choice_reward_results[model_name] = evaluate_llm(model, tokenizer, ibl_all_mouse_name, ibl_choice_reward, ibl_choice_reward_eval_loc, list(ibl_curr_previous_choice_mapping.values()))
        ibl_stim_choice_reward_results[model_name] = evaluate_llm(model, tokenizer, ibl_all_mouse_name, ibl_stim_choice_reward, ibl_stim_choice_reward_eval_loc, list(ibl_curr_previous_choice_mapping.values()))
        
        rew_learn_side_only_results[model_name] = evaluate_llm(model, tokenizer, rew_learn_all_mouse_name, rew_learn_side_only, rew_learn_side_only_eval_loc, list(rew_learn_curr_side_mapping.values()))
        rew_learn_side_reward_results[model_name] = evaluate_llm(model, tokenizer, rew_learn_all_mouse_name, rew_learn_side_reward, rew_learn_side_reward_eval_loc, list(rew_learn_curr_side_mapping.values()))
        del model

    # save results
    with open(os.path.join(RESULTS_PATH, 'ibl_results.json'), 'w') as f:
        json.dump({
            'choice_only_results': ibl_choice_only_results,
            'stim_choice_results': ibl_stim_choice_results,
            'choice_reward_results': ibl_choice_reward_results,
            'stim_choice_reward_results': ibl_stim_choice_reward_results
        }, f, indent=4)
    with open(os.path.join(RESULTS_PATH, 'rew_learn_results.json'), 'w') as f:
            json.dump({
                'side_only_results': rew_learn_side_only_results,
                'side_reward_results': rew_learn_side_reward_results,
            }, f, indent=4)

if __name__ == "__main__":
    main()