from tqdm import tqdm
import time
import random
import os
from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor

import google.generativeai as genai


########################Gemini########################
class GeminiJudgeText2Image:
    def __init__(self, args):
        self.args = args
        self.JUDGE = "gemini-1.5-pro-latest"  # "gemini-1.5-flash-latest"
        self.FIX_INTERVAL_SECOND = 0
        self.MAX_RETRY_NUM = 999999
        self.MAX_NEW_TOKENS = 999
        
        self.FORMAT_MAXRETRY = 10

        load_dotenv()
        genai.configure(api_key=os.getenv('k_g'))   # set the api key here
        self.model = genai.GenerativeModel(self.JUDGE)
        
        self.safety_settings={
            'harm_category_harassment':'block_none',
            'harm_category_hate_speech': 'block_none',
            'harm_category_sexually_explicit': 'block_none',
            'harm_category_dangerous_content': 'block_none'
            }

    def format_prompts(self, inputs):
        
        # input format logic here
        
        return formated
    
    def _GPT_decode(self, inputs):
        completion = self.model.generate_content(
                            self.format_prompts(inputs),
                            generation_config=genai.types.GenerationConfig(
                                candidate_count=1,
                                max_output_tokens=self.MAX_NEW_TOKENS,
                                ),
                            safety_settings=self.safety_settings,
                        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion.text


    def GPT_decode(self, inputs):
        delay = 1
        blocked = 0
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._GPT_decode(inputs)
                return response_content
            except Exception as e:
                if 'quick accessor' in str(e) or 'block' in str(e):
                    print("Content blocked, retrying ...")
                    blocked += 1
                    if blocked > 10:
                        print("Blocked for too many times, using 'Response not available "
                              "due to content restrictions.' as response, exiting...")
                        return 'Response not available due to content restrictions.'
                elif 'quota' in str(e).lower() or 'limit' in str(e).lower():
                    exponential_base = 2
                    delay *= exponential_base * (1 + random.random())
                    print(f"Error, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                    print(e)
                    time.sleep(delay)
                    continue
                else:
                    print(f"Error in decode, retrying...")
                    print(e)
                    time.sleep(10)
                    continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return 'Error'


    def annotate_p(self, task):    
        # the annotation logic here
        
        pass


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