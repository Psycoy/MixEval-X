from tqdm import tqdm
import time
import random
import os
from dotenv import load_dotenv
import base64
import re
import ast

from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from openai._exceptions import RateLimitError, BadRequestError
from httpx import Timeout

from mixeval_x.prompts.judge_prompts import (
    text2action_gpt_judge,
    )

########################ChatGPT########################
class ChatGPTJudgeText2Action:
    def __init__(self, args):
        self.args = args
        self.JUDGE = args.judge_model
        self.FIX_INTERVAL_SECOND = 0
        self.MAX_RETRY_NUM = 999999
        self.MAX_NEW_TOKENS = 999
        
        self.FORMAT_MAXRETRY = 10

        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv('MODEL_PARSER_API'),
            timeout=Timeout(timeout=60.0, connect=5.0)
        )
        
    @staticmethod
    def get_score_from_judge(judge_response):
        """
        Get the score from the judge response.
        """
        one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
        one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
        
        match = re.search(one_score_pattern, judge_response)
        if not match:
            match = re.search(one_score_pattern_backup, judge_response)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
            
        return float(rating)

    def format_prompts(self, inputs, mode, history=[]):
        
        if mode == 'turn_1':
            formated = text2action_gpt_judge(*inputs)
        elif mode == 'append_message' and history:
            history.append(inputs)
            formated = history
        else:
            raise ValueError(f"Invalid mode: {mode}.")
        
        return formated
    
    def _GPT_decode(self, inputs, mode, history=[]):
        completion = self.client.chat.completions.create(
                            model=self.JUDGE,
                            response_format={ "type": 'text'},
                            max_tokens=self.MAX_NEW_TOKENS,
                            messages=self.format_prompts(inputs, mode, history),
                            )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion

    def GPT_decode(self, inputs, mode, history=[]):
        delay = 1
        blocked = 0
        for i in range(self.MAX_RETRY_NUM):
            try:
                completion = self._GPT_decode(inputs, mode, history)
                return completion
            except RateLimitError as e:
                exponential_base = 2
                delay *= exponential_base * (1 + random.random())
                print(f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                print(e)
                time.sleep(delay)
                continue
            except BadRequestError as e:
                blocked += 1
                if blocked >= 10:
                    print("Blocked too many times, skipping...")
                    return 'Blocked'
                print(f"Input is blocked, retrying...")
                print(e)
                time.sleep(1)
                continue
            except Exception as e:
                print(f"Error in GPT_decode, retrying...")
                print(e)
                time.sleep(1)
                continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return 'Error'

    def annotate_p(self, task):    
        task_description = task['task description']
        allowed_actions = task['allowed actions']
        visible_objects = task['visible objects']
        already_executed_steps = task['already executed steps']
        target = task['target']
        model_response = task['response']
        
        # first turn
        inputs = (task_description, allowed_actions, visible_objects, already_executed_steps, target, model_response)
        
        completion = self.GPT_decode(inputs, 'turn_1')
        if completion == 'Error':
            print(f"Error in GPT_decode, the entry treated as bad entry.")
            task['judge_response'] = '[[0.0]]'
        elif completion == 'Blocked':
            print(f"{task}: \n\nBlocked, the entry treated as bad entry.")
            task['judge_response'] = '[[0.0]]'
        else:
            annotation = completion.choices[0].message.content
            task['judge_response'] = annotation
            for i in range(self.FORMAT_MAXRETRY):
                if self.get_score_from_judge(annotation) != -1:
                    task['judge_score'] = self.get_score_from_judge(annotation)
                    break
                else:
                    print(f"No score found in the response, retrying...")
                    history = text2action_gpt_judge(*inputs)
                    completion = self.GPT_decode({"role": "user", "content": "Continue your judgment and finish by outputting a final score with the above-mentioned format."}, 'append_message', history)
                    annotation = completion.choices[0].message.content

            if self.get_score_from_judge(annotation) != -1:
                task['judge_score'] = self.get_score_from_judge(annotation)
            else:
                task['judge_score'] = None
                print(f"No score found in the response, please inspect and retry.")
        
        
        return task

    def annotate_parallel(self, tasks):
        print(f"Parsing in parallel, in total {self.args.api_parallel_num} threads.")
        results = []
        with ThreadPoolExecutor(self.args.api_parallel_num) as executor:
            for entry in tqdm(
                executor.map(self.annotate_p, tasks), total=len(tasks)
            ):
                results.append(entry)
        if None in results:
            raise ValueError("Some entries are not annotated due to errors in annotate_p, please inspect and retry.")
        return results

########################Claude 3########################
class ClaudeJudgeText2Action:
    def __init__(self):
        raise NotImplementedError
    

########################Gemini########################
class GeminiJudgeText2Action:
    def __init__(self):
        raise NotImplementedError