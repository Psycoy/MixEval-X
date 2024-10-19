'''
Usage:
python -m mixeval_x.compute_metrics_mmgen \
    --benchmark mixeval_x_text2action \
    --judge_model "gpt-4o-2024-08-06" \
    --models_to_eval \
        llama_3_2_90b
'''
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
    eval_text2image,
    eval_text2action,
    eval_image2action
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", 
        type=str, 
        choices=["mixeval_x_text2image", 
                 "mixeval_x_text2video", 
                 "mixeval_x_text2audio", 
                 "mixeval_x_text2action",
                 "mixeval_x_image2action"
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
        "--image2action_image_dir", 
        type=str, 
        default="mixeval_x/data/image2action/image2action_closeended/images/", 
        help="Path to the images of image2action data."
        )
    parser.add_argument(
        "--image2action_example_image_path", 
        type=str, 
        default="mixeval_x/utils/misc/example_image.jpg", 
        help="Path to the example image of the image2action split."
        )
    parser.add_argument(
        "--judge_model",
        type=str, 
        default="gpt-4o", 
        help="Judge model for text2image score computation."
        )
    parser.add_argument(
        "--models_to_eval", 
        nargs='+',
        default=None, 
        help="Models to evaluate."
        )
    parser.add_argument(
        "--models_to_ignore", 
        nargs='+',
        default=None, 
        help="Models that would be ignored for free-form."
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


def compute_metric_text2image(args):
    
    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        
    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
    
    for model in models:
        print(f"Parsing model: {model}")
        
        if args.models_to_ignore is not None and model in args.models_to_ignore:
            print(f"Model {model} is ignored for text2image.")
            continue
        
        if args.compute_score_from_judged_file:
            results = []
            judge_file = os.path.join(
                args.model_response_dir,
                args.benchmark,
                model,
                f"judge_results_t2i_model_judge_{args.judge_model}.jsonl"
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
                f"{model}_t2i.jsonl"
                )
            tasks = []
            with open(ans_file, "r") as f:
                for line in f:
                    ans_dict = json.loads(line)
                    tasks.append(ans_dict)
            results = eval_text2image(args, tasks)
        
        score_dict_model = {}
        for judge_dict in results:
            judge_score = judge_dict["judge_score_1st_turn"]
            if '1st_turn' not in score_dict_model:
                score_dict_model['1st_turn'] = []
            score_dict_model['1st_turn'].append(judge_score)
            judge_score = judge_dict["judge_score_2nd_turn"]
            if '2nd_turn' not in score_dict_model:
                score_dict_model['2nd_turn'] = []
            score_dict_model['2nd_turn'].append(judge_score)
            
        for key, value in score_dict_model.items():
            score_dict_model[key] = round(sum(value)/len(value), 3)
        score_dict[model] = score_dict_model

        with open(os.path.join(
            args.model_response_dir, 
            args.benchmark,
            model,
            f"judge_results_t2i_model_judge_{args.judge_model}.jsonl"), "w") as f:
            for case in results:
                f.write(json.dumps(case) + "\n")
        
        if not args.compute_score_from_judged_file:
            print("Sleep 20 seconds to avoid ratelimit error ... ")
            time.sleep(20)
    
    if args.verbose:
        print(f"[Text2Image]")
        for model, score in score_dict.items():
            print(f"{model}: {json.dumps(score, indent=4)}")
    
    # write score_dict
    score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1]['1st_turn'], reverse=True))
    with open(os.path.join(
        args.model_response_dir, 
        args.benchmark,
        "score_t2i.json"), "w") as f:
        f.write(json.dumps(score_dict, indent=4) + "\n")
        
    return score_dict

def compute_metric_text2action(args):
    
    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        
    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
    
    for model in models:
        print(f"Parsing model: {model}")
        
        if args.models_to_ignore is not None and model in args.models_to_ignore:
            print(f"Model {model} is ignored for text2action.")
            continue
        
        if args.compute_score_from_judged_file:
            results = []
            judge_file = os.path.join(
                args.model_response_dir,
                args.benchmark,
                model,
                f"judge_results_t2a_model_judge_{args.judge_model}.jsonl"
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
                f"{model}_t2a.jsonl"
                )
            tasks = []
            with open(ans_file, "r") as f:
                for line in f:
                    ans_dict = json.loads(line)
                    tasks.append(ans_dict)
            results = eval_text2action(args, tasks)
        
        score_dict_model = {}
        for judge_dict in results:
            judge_score = judge_dict["judge_score"]
            if 'overall' not in score_dict_model:
                score_dict_model['overall'] = []
            score_dict_model['overall'].append(judge_score)
            
        for key, value in score_dict_model.items():
            score_dict_model[key] = round(sum(value)/len(value), 3)
        score_dict[model] = score_dict_model

        with open(os.path.join(
            args.model_response_dir, 
            args.benchmark,
            model,
            f"judge_results_t2a_model_judge_{args.judge_model}.jsonl"), "w") as f:
            for case in results:
                f.write(json.dumps(case) + "\n")
        
        if not args.compute_score_from_judged_file:
            print("Sleep 20 seconds to avoid ratelimit error ... ")
            time.sleep(20)
    
    if args.verbose:
        print(f"[Text2Action]")
        for model, score in score_dict.items():
            print(f"{model}: {json.dumps(score, indent=4)}")
    
    # sort and write score_dict
    score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1]['overall'], reverse=True))
    with open(os.path.join(
        args.model_response_dir, 
        args.benchmark,
        "score_t2a.json"), "w") as f:
        f.write(json.dumps(score_dict, indent=4) + "\n")
        
    return score_dict


def compute_metric_image2action(args):
    
    score_dict = {}
    if args.models_to_eval is not None:
        models = args.models_to_eval
        
    else:
        if os.path.exists(args.model_response_dir):
            models = os.listdir(args.model_response_dir)
    
    for model in models:
        print(f"Parsing model: {model}")
        
        if args.models_to_ignore is not None and model in args.models_to_ignore:
            print(f"Model {model} is ignored for image2action.")
            continue
        
        if args.compute_score_from_judged_file:
            results = []
            judge_file = os.path.join(
                args.model_response_dir,
                args.benchmark,
                model,
                f"judge_results_i2a_model_judge_{args.judge_model}.jsonl"
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
                f"{model}_i2a.jsonl"
                )
            tasks = []
            with open(ans_file, "r") as f:
                for line in f:
                    ans_dict = json.loads(line)
                    tasks.append(ans_dict)
            results = eval_image2action(args, tasks)
        
        score_dict_model = {}
        for judge_dict in results:
            judge_score = judge_dict["judge_score"]
            if 'overall' not in score_dict_model:
                score_dict_model['overall'] = []
            score_dict_model['overall'].append(judge_score)
            
        for key, value in score_dict_model.items():
            score_dict_model[key] = round(sum(value)/len(value), 3)
        score_dict[model] = score_dict_model

        with open(os.path.join(
            args.model_response_dir, 
            args.benchmark,
            model,
            f"judge_results_i2a_model_judge_{args.judge_model}.jsonl"), "w") as f:
            for case in results:
                f.write(json.dumps(case) + "\n")
        
        if not args.compute_score_from_judged_file:
            print("Sleep 20 seconds to avoid ratelimit error ... ")
            time.sleep(20)
    
    if args.verbose:
        print(f"[Image2Action]")
        for model, score in score_dict.items():
            print(f"{model}: {json.dumps(score, indent=4)}")
    
    # sort and write score_dict
    score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1]['overall'], reverse=True))
    with open(os.path.join(
        args.model_response_dir, 
        args.benchmark,
        "score_i2a.json"), "w") as f:
        f.write(json.dumps(score_dict, indent=4) + "\n")
        
    return score_dict

def compute_metric(args):
    if args.benchmark == "mixeval_x_text2image":
        compute_metric_text2image(args)
    elif args.benchmark == "mixeval_x_text2video":
        raise NotImplementedError("Benchmark not implemented yet.")
    elif args.benchmark == "mixeval_x_text2audio":
        raise NotImplementedError("Benchmark not implemented yet.")
    elif args.benchmark == "mixeval_x_text2action":
        compute_metric_text2action(args)
    elif args.benchmark == "mixeval_x_image2action":
        compute_metric_image2action(args)
    else:
        raise ValueError(f"Invalid benchmark: {args.benchmark}")
            

if __name__ == '__main__':
    set_seed()
    args = parse_args()
    compute_metric(args)