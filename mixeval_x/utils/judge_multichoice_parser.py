from tqdm import tqdm
import time
import random
import os
from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from openai._exceptions import RateLimitError, BadRequestError
from httpx import Timeout

from mixeval_x.prompts.judge_prompts import (
    image2text_gpt_judge_for_closeended_multiplechoice,
    video2text_gpt_judge_for_closeended_multiplechoice,
    )
from mixeval_x.utils.common_utils import extract_basemodel_response_2e

########################ChatGPT########################
class ChatGPTJudgeCloseendMultichoice:
    def __init__(self, args):
        self.args = args
        
        self.JUDGE = args.multichoice_judge
        self.FIX_INTERVAL_SECOND = 0
        self.MAX_RETRY_NUM = 99
        self.MAX_NEW_TOKENS = 999

        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv('MODEL_PARSER_API'),
            timeout=Timeout(timeout=20.0, connect=5.0)
        )

    def format_prompts(self, inputs):
        prompt, options, response = inputs
        option_letters = [chr(ord("A") + i) for i in range(len(options))]
        options = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
        if self.args.benchmark in ['image2text', 'image2text_hard']:
            formated = image2text_gpt_judge_for_closeended_multiplechoice(prompt, options, response)
        elif self.args.benchmark in ['video2text', 'video2text_hard']:
            formated = video2text_gpt_judge_for_closeended_multiplechoice(prompt, options, response)
        else:
            raise ValueError(f"Benchmark {self.args.benchmark} not supported in the closeended parser.")
        return formated
    
    def _GPT_decode(self, inputs):
        completion = self.client.chat.completions.create(
                            model=self.JUDGE,
                            response_format={ "type": 'text'},
                            max_tokens=self.MAX_NEW_TOKENS,
                            messages=self.format_prompts(inputs),
                            )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion


    def GPT_decode(self, inputs):
        delay = 1
        blocked = 0
        for i in range(self.MAX_RETRY_NUM):
            try:
                completion = self._GPT_decode(inputs)
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
        prompt = task['prompt']
        options = task['options']
        response = task['response']
        
        if hasattr(self.args, 'model_type'):
            if self.args.model_type == 'BaseModel':
                response = extract_basemodel_response_2e(response)
                task['response_extracted'] = response
            elif self.args.model_type == 'ChatModel':
                pass
            elif self.args.model_type == 'APIModelBase':
                pass
            else:
                raise ValueError(f"Model type {self.args.model_type} not supported.")
            
        if not isinstance(options, list):
            print(f"Invalid target: {options}")
            return None
        
        inputs = (prompt, options, response)
        
        completion = self.GPT_decode(inputs)
        if completion == 'Error':
            print(f"Error in GPT_decode, the entry {task} will be retried later...")
            task['judge_response'] = None
            return task
        elif completion == 'Blocked':
            print(f"{task}: \n\nBlocked, the entry treated as bad entry. Randomly assigning a choice.")
            options = task['options']
            option_letters = [chr(ord("A") + i) for i in range(len(options))]
            task['judge_response'] = f"[[{random.choice(option_letters)}]]"
            return task
        annotation = completion.choices[0].message.content
        task['judge_response'] = annotation
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
class ClaudeJudgeCloseendMultichoice:
    def __init__(self):
        raise NotImplementedError
    

########################Gemini########################
class GeminiJudgeCloseendMultichoice:
    def __init__(self):
        raise NotImplementedError