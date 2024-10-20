import json
import argparse
import os
from tqdm import tqdm
import time
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

from mixeval_x.utils.common_utils import set_seed
from mixeval_x.utils.metric_utils import (
    parse_multi_choice_response_rule,
    parse_multi_choice_response_model,
    eval_multi_choice,
    eval_freeform_model,
    parse_freeform_response_rule,
    eval_freeform_rule,
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", 
        type=str, 
        choices=["image2text", 
                 "video2text", 
                 "audio2text", 
                 "image2text_hard", 
                 "video2text_hard", 
                 "audio2text_hard"
                 ], 
        required=True,
        help="Benchmark to evaluate."
        )
    parser.add_argument(
        "--model_response_dir", 
        type=str, 
        default="mixeval_x/data/image2text/close_ended/model_responses/", 
        help="Path to model responses."
        )
    parser.add_argument(
        "--multichoice_judge",
        type=str, 
        default="gpt-3.5-turbo-0125", 
        help="Judge model for multiple-choice score computation."
        )
    parser.add_argument(
        "--freeform_judge",
        type=str, 
        default="gpt-3.5-turbo-0125", 
        help="Judge model for freeform score computation."
        )
    parser.add_argument(
        "--models_to_eval", 
        nargs='+',
        default=None, 
        help="Models to evaluate."
        )
    parser.add_argument(
        "--models_to_ignore_ff", 
        nargs='+',
        default=None, 
        help="Models that would be ignored for free-form."
        )
    parser.add_argument(
        "--models_to_ignore_mp", 
        nargs='+',
        default=None, 
        help="Models that would be ignored for multiple-choice."
        )
    parser.add_argument(
        "--free_form_parser", 
        type=str, 
        default="model", 
        choices=["model", "rule"], 
        help="Parser for freeform responses, either model parser or rule-based parser.")
    parser.add_argument(
        "--multi_choice_parser", 
        type=str, 
        default="model", 
        choices=["model", "rule"], 
        help="Parser for multiple-choice responses, either model parser or rule-based parser."
        )
    parser.add_argument(
        "--api_parallel_num", 
        type=int, 
        default=100, 
        help="Number of parallel threads for calling the model parser api if use model parsing." 
        "If you hit rate limit error frequently, try to reduce this number."
        )
    parser.add_argument(
        "--compute_score_from_judged_file", 
        action="store_true", 
        help="Whether to compute score directly from the judged file."
        "This will save budge for those models that has been judged before."
        "it also helps to do some analysis easily without running judgements again."
        )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Print verbose information."
        )
    return parser.parse_args()


def compute_metric_closeended_freeform_modelparse(args):
    
    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        
    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
    
    for model in models:
        print(f"Parsing model: {model}")
        
        if args.models_to_ignore_ff is not None and model in args.models_to_ignore_ff:
            print(f"Model {model} is ignored for free-form.")
            continue
        
        if args.compute_score_from_judged_file:
            results = []
            judge_file = os.path.join(
                args.model_response_dir, 
                args.benchmark,
                model,
                f"judge_results_ff_model_judge_{args.freeform_judge}.jsonl"
                )
            with open(judge_file, "r") as f:
                for line in f:
                    judge_dict = json.loads(line)
                    results.append(judge_dict)
        else:
            ans_file = os.path.join(
                args.model_response_dir, 
                args.benchmark,
                model,
                f"{model}_ff.jsonl"
                )
            tasks = []
            with open(ans_file, "r") as f:
                for line in f:
                    ans_dict = json.loads(line)
                    tasks.append(ans_dict)
            results = eval_freeform_model(args, tasks)
        
        score_dict_model = {}
        for judge_dict in results:
            judge_score = judge_dict["judge_score"]
            if 'overall' not in score_dict_model:
                score_dict_model['overall'] = []
            score_dict_model['overall'].append(judge_score)
            if judge_dict['benchmark_name'] not in score_dict_model:
                score_dict_model[judge_dict['benchmark_name']] = []
            score_dict_model[judge_dict['benchmark_name']].append(judge_score)
            
        for key, value in score_dict_model.items():
            score_dict_model[key] = round(sum(value)/len(value), 3)
        score_dict[model] = score_dict_model

        with open(os.path.join(
            args.model_response_dir, 
            args.benchmark,
            model,
            f"judge_results_ff_model_judge_{args.freeform_judge}.jsonl"), "w") as f:
            for case in results:
                f.write(json.dumps(case) + "\n")
        
        if not args.compute_score_from_judged_file:
            print("Sleep 60 seconds to avoid ratelimit error ... ")
            time.sleep(60)
    
    if args.verbose:
        print(f"[Close-ended Free-form Model Parser]")
        for model, score in score_dict.items():
            print(f"{model}: {json.dumps(score, indent=4)}")
    
    # write score_dict
    score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1]['overall'], reverse=True))
    with open(os.path.join(
        args.model_response_dir, 
        args.benchmark,
        "score_ff.json"), "w") as f:
        f.write(json.dumps(score_dict, indent=4) + "\n")
    
    # print(f"Number of ff entries: {len(results)}")
    return score_dict, len(results)

def compute_metric_closeended_multichoice_modelparse(args):

    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        
    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
        
    for model in models:
        print(f"Parsing model: {model}")
        
        if args.models_to_ignore_mp is not None and model in args.models_to_ignore_mp:
            print(f"Model {model} is ignored for multiple-choice.")
            continue

        if args.compute_score_from_judged_file:
            results = []
            judge_file = os.path.join(
                args.model_response_dir, 
                args.benchmark,
                model,
                f"judge_results_mp_model_judge_{args.multichoice_judge}.jsonl"
                )
            with open(judge_file, "r") as f:
                for line in f:
                    judge_dict = json.loads(line)
                    results.append(judge_dict)
        else:
            ans_file = os.path.join(
                    args.model_response_dir, 
                    args.benchmark,
                    model,
                    f"{model}_mp.jsonl"
                    )
            with open(ans_file, "r") as f:
                ans_dicts = []
                for line in f:
                    ans_dict = json.loads(line)
                    ans_dicts.append(ans_dict)
                    
            results = parse_multi_choice_response_model(args, ans_dicts)
        
        score_dict_model = {}
        for judge_dict in results:
            options = judge_dict["options"]
            target = judge_dict["target"]
            assert isinstance(target, list) and len(target) == 1, \
                f"Invalid target: {target}"
            all_choices = [chr(ord("A") + i) for i in range(len(options))]
            model_choice = judge_dict['judge_option']
            target_id = all_choices[target[0]]
            judge_score = 1 if eval_multi_choice(target_id, model_choice) else 0
            
            # add score
            if 'overall' not in score_dict_model:
                score_dict_model['overall'] = []
            score_dict_model['overall'].append(judge_score)
            if judge_dict['benchmark_name'] not in score_dict_model:
                score_dict_model[judge_dict['benchmark_name']] = []
            score_dict_model[judge_dict['benchmark_name']].append(judge_score)
            
        for key, value in score_dict_model.items():
            score_dict_model[key] = round(sum(value)/len(value), 3)
        score_dict[model] = score_dict_model
        
        with open(os.path.join(
                        args.model_response_dir, 
                        args.benchmark,
                        model,
                        f"judge_results_mp_model_judge_{args.multichoice_judge}.jsonl"
                               ), "w") as f:
            for case in results:
                f.write(json.dumps(case) + "\n")
                
        if not args.compute_score_from_judged_file:
            print("Sleep 60 seconds to avoid ratelimit error ... ")
            time.sleep(60)
    
    if args.verbose:
        print(f"[Close-ended Multiple-choice Model Parser]")
        for model, score in score_dict.items():
            print(f"{model}: {json.dumps(score, indent=4)}")
            
    # write score_dict
    score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1]['overall'], reverse=True))
    with open(os.path.join(
        args.model_response_dir, 
        args.benchmark,
        "score_mp.json"), "w") as f:
        f.write(json.dumps(score_dict, indent=4) + "\n")
    
    # print(f"Number of mp entries: {len(results)}")
    return score_dict, len(results)

def compute_metric_closeended_freeform(args):
    return compute_metric_closeended_freeform_modelparse(args)

def compute_metric_closeended_multichoice(args):
    return compute_metric_closeended_multichoice_modelparse(args)

def compute_metric_closeended(args):
    if "audio" not in args.benchmark:
        score_dict_ff, ff_num = compute_metric_closeended_freeform(args)
        score_dict_mp, mp_num = compute_metric_closeended_multichoice(args)
        
        models_ff = set(score_dict_ff.keys())
        models_mp = set(score_dict_mp.keys())
        common_models = models_ff.intersection(models_mp)
        missing_models = models_ff.union(models_mp) - common_models
        if missing_models:
            print(f"Something went wrong when computing the free-form or multiple-choice "
                f"split of these models: \n{missing_models}\n\nA possible reason may be that they lack a model answer file. "
                "Skipping them...")
        
        score_dict = {}
        for model in common_models:
            score_dir = os.path.join(
                args.model_response_dir, 
                args.benchmark,
                model,
                )
            score_dict_model = {
                "overall score (final score)": (score_dict_ff[model]['overall']*ff_num + score_dict_mp[model]['overall']*mp_num) / (ff_num + mp_num),
                **{f"{k} (free-form)":v for k, v in score_dict_ff[model].items() if k != "overall"},
                **{f"{k} (multiple-choice)":v for k, v in score_dict_mp[model].items() if k != "overall"},
                }
            score_dict[model] = score_dict_model
            with open(os.path.join(score_dir, "score.json"), "w") as f:
                f.write(json.dumps(score_dict_model, indent=4) + "\n")
        
        # sort and write score_dict
        score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1]['overall score (final score)'], reverse=True))
        with open(os.path.join(args.model_response_dir, args.benchmark, "score.json"), "w") as f:
            f.write(json.dumps(score_dict, indent=4) + "\n")
    else:
        # only takes the freeform score
        score_dict_ff = compute_metric_closeended_freeform(args)
        models_ff = set(score_dict_ff.keys())
        for model in models_ff:
            score_dir = os.path.join(
                args.model_response_dir, 
                args.benchmark,
                model,
                )
            score_dict_model = {
                "overall score (final score)": score_dict_ff[model]['overall'],
                **{k:v for k, v in score_dict_ff[model].items() if k != "overall"},
                }
            with open(os.path.join(score_dir, "score.json"), "w") as f:
                f.write(json.dumps(score_dict_model, indent=4) + "\n")
                
        # sort and write score_dict
        score_dict_ff = dict(sorted(score_dict_ff.items(), key=lambda x: x[1]['overall'], reverse=True))
        with open(os.path.join(args.model_response_dir, args.benchmark, "score.json"), "w") as f:
            f.write(json.dumps(score_dict_ff, indent=4) + "\n")
            
            

def compute_metric(args):
    if args.benchmark in ["image2text", "video2text", "audio2text", "image2text_hard", "video2text_hard", "audio2text_hard"]:
        compute_metric_closeended(args)
    else:
        raise ValueError(f"Invalid benchmark: {args.benchmark}, please choose from "
                         f"['image2text', 'video2text', 'audio2text', 'image2text_hard', 'video2text_hard', 'audio2text_hard']")
            

if __name__ == '__main__':
    set_seed()
    args = parse_args()
    compute_metric(args)