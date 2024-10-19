from tqdm import tqdm
import time
import random
import os
from dotenv import load_dotenv
import base64
import PIL.Image
from PIL import Image
import re
import ast

from concurrent.futures import ThreadPoolExecutor
from httpx import Timeout

from openai import OpenAI
from openai._exceptions import RateLimitError as RateLimitError_openai, BadRequestError as BadRequestError_openai

import anthropic
from anthropic._exceptions import RateLimitError as RateLimitError_anthropic

import google.generativeai as genai

from mixeval_x.prompts.judge_prompts import (
    text2image_gpt_judge_turn1,
    text2image_gpt_judge_turn2,
    text2image_claude_judge_turn1,
    text2image_claude_judge_turn2,
    text2image_gemini_judge_turn1,
    text2image_gemini_judge_turn2,
    )

########################ChatGPT########################
class ChatGPTJudgeText2Image:
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
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
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
            formated = text2image_gpt_judge_turn1(*inputs)
        elif mode == 'turn_2':
            formated = text2image_gpt_judge_turn2(*inputs)
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
            except RateLimitError_openai as e:
                exponential_base = 2
                delay *= exponential_base * (1 + random.random())
                print(f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                print(e)
                time.sleep(delay)
                continue
            except BadRequestError_openai as e:
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
        prompt1 = task['first_turn_user_prompt']
        prompt2 = task['second_turn_user_prompt']
        
        if not os.path.exists(task['gen_1st_turn']):
            task['judge_response_1st_turn'] = 'Image not found, treated as a neutral entry.'
            task['judge_score_1st_turn'] = 5.0
            task['judge_response_2nd_turn'] = 'Image not found, treated as a neutral entry.'
            task['judge_score_2nd_turn'] = 5.0
            print(f"Image 1 not found, treated as a neutral entry.")
            return task
        else:
            image1 = self.encode_image(task['gen_1st_turn'])
            
            # first turn
            inputs = (prompt1, image1)
            
            completion = self.GPT_decode(inputs, 'turn_1')
            if completion == 'Error':
                print(f"Error in GPT_decode, the entry treated as bad entry.")
                task['judge_response_1st_turn'] = '[[0.0]]'
            elif completion == 'Blocked':
                print(f"{task}: \n\nBlocked, the entry treated as bad entry.")
                task['judge_response_1st_turn'] = '[[0.0]]'
            else:
                annotation = completion.choices[0].message.content
                task['judge_response_1st_turn'] = annotation
                for i in range(self.FORMAT_MAXRETRY):
                    if self.get_score_from_judge(annotation) != -1:
                        task['judge_score_1st_turn'] = self.get_score_from_judge(annotation)
                        break
                    else:
                        print(f"No score found in the response, retrying...")
                        history = text2image_gpt_judge_turn1(*inputs)
                        history.append({"role": "assistant", "content": annotation})
                        completion = self.GPT_decode({"role": "user", "content": "Continue your judgment and finish by outputting a final rating with the above-mentioned format, i.e., the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Your rating: "}, 'append_message', history)
                        annotation = completion.choices[0].message.content
                        history.append({"role": "assistant", "content": annotation})

                if self.get_score_from_judge(annotation) != -1:
                    task['judge_score_1st_turn'] = self.get_score_from_judge(annotation)
                else:
                    task['judge_score_1st_turn'] = 5.0
                    print(f"No score found in the response, please inspect and retry.")
        
        if not os.path.exists(task['gen_2nd_turn']):
            task['judge_response_2nd_turn'] = 'Image not found, treated as a neutral entry.'
            task['judge_score_2nd_turn'] = 5.0
            print(f"Image 2 not found, treated as a neutral entry.")
            return task
        else:
            image2 = self.encode_image(task['gen_2nd_turn'])
            # second turn
            inputs = (image1, prompt2, image2)
            completion = self.GPT_decode(inputs, 'turn_2')
            if completion == 'Error':
                print(f"Error in GPT_decode, the entry treated as bad entry.")
                task['judge_response_2nd_turn'] = '[[0.0]]'
            elif completion == 'Blocked':
                print(f"{task}: \n\nBlocked, the entry treated as bad entry.")
                task['judge_response_2nd_turn'] = '[[0.0]]'
            else:
                annotation = completion.choices[0].message.content
                task['judge_response_2nd_turn'] = annotation
                for i in range(self.FORMAT_MAXRETRY):
                    if self.get_score_from_judge(annotation) != -1:
                        task['judge_score_2nd_turn'] = self.get_score_from_judge(annotation)
                        break
                    else:
                        print(f"No score found in the response, retrying...")
                        history = text2image_gpt_judge_turn1.format(*inputs)
                        history.append({"role": "assistant", "content": annotation})
                        completion = self.GPT_decode({"role": "user", "content": "Continue your judgment and finish by outputting a final rating with the above-mentioned format, i.e., the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Your rating: "}, 'append_message', history)
                        annotation = completion.choices[0].message.content
                        history.append({"role": "assistant", "content": annotation})

                if self.get_score_from_judge(annotation) != -1:
                    task['judge_score_2nd_turn'] = self.get_score_from_judge(annotation)
                else:
                    task['judge_score_2nd_turn'] = 5.0
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
class ClaudeJudgeText2Image:
    def __init__(self, args):
        self.args = args
        self.JUDGE = args.judge_model
        self.FIX_INTERVAL_SECOND = 0
        self.MAX_RETRY_NUM = 999999
        self.MAX_NEW_TOKENS = 999
        
        self.FORMAT_MAXRETRY = 10

        load_dotenv()
        self.client = anthropic.Anthropic(
            api_key=os.getenv('k_ant'),
            timeout=Timeout(timeout=20.0, connect=5.0)
        )
    
    @staticmethod
    def encode_image(image_path):
        
        def convert_image_to_jpeg(image_path, output_path):
            # Open the image and convert it to JPEG to ensure format compliance
            img = Image.open(image_path)
            img.convert("RGB").save(output_path, format="JPEG")
        convert_image_to_jpeg(image_path, image_path)
        
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
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
            formated = text2image_claude_judge_turn1(*inputs)
        elif mode == 'turn_2':
            formated = text2image_claude_judge_turn2(*inputs)
        elif mode == 'append_message' and history:
            history.append(inputs)
            formated = history
        else:
            raise ValueError(f"Invalid mode: {mode}.")
        
        return formated
    
    def _GPT_decode(self, inputs, mode, history=[]):
        completion = self.client.messages.create(
                            model=self.JUDGE,
                            max_tokens=self.MAX_NEW_TOKENS,
                            messages=self.format_prompts(inputs, mode, history),
                            )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion


    def GPT_decode(self, inputs, mode, history=[]):
        delay = 1
        for i in range(self.MAX_RETRY_NUM):
            try:
                completion = self._GPT_decode(inputs, mode, history)
                return completion
            except RateLimitError_anthropic as e:
                exponential_base = 2
                delay *= exponential_base * (1 + random.random())
                print(f"RateLimitError, retrying after {round(delay, 2)} seconds, {i+1}-th retry...")
                print(e)
                time.sleep(delay)
                continue
            except Exception as e:
                print(f"Error in decode, retrying...")
                print(e)
                time.sleep(1)
                continue
        print(f"Failed after {self.MAX_RETRY_NUM} retries.")
        return 'Error'


    def annotate_p(self, task):    
        prompt1 = task['first_turn_user_prompt']
        prompt2 = task['second_turn_user_prompt']
        
        if not os.path.exists(task['gen_1st_turn']):
            task['judge_response_1st_turn'] = 'Image not found, treated as a neutral entry.'
            task['judge_score_1st_turn'] = 5.0
            task['judge_response_2nd_turn'] = 'Image not found, treated as a neutral entry.'
            task['judge_score_2nd_turn'] = 5.0
            print(f"Image 1 not found, treated as a neutral entry.")
            return task
        else:
            image1 = self.encode_image(task['gen_1st_turn'])
            
            # first turn
            inputs = (prompt1, image1)
            
            completion = self.GPT_decode(inputs, 'turn_1')
            if completion == 'Error':
                print(f"Error in GPT_decode, the entry treated as bad entry.")
                task['judge_response_1st_turn'] = '[[0.0]]'
            elif completion == 'Blocked':
                print(f"{task}: \n\nBlocked, the entry treated as bad entry.")
                task['judge_response_1st_turn'] = '[[0.0]]'
            else:
                annotation = completion.content[0].text
                task['judge_response_1st_turn'] = annotation
                for i in range(self.FORMAT_MAXRETRY):
                    if self.get_score_from_judge(annotation) != -1:
                        task['judge_score_1st_turn'] = self.get_score_from_judge(annotation)
                        break
                    else:
                        print(f"No score found in the response, retrying...")
                        history = text2image_claude_judge_turn1(*inputs)
                        history.append({"role": "assistant", "content": annotation})
                        completion = self.GPT_decode({"role": "user", "content": "Continue your judgment and finish by outputting a final rating with the above-mentioned format, i.e., the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Your rating: "}, 'append_message', history)
                        annotation = completion.content[0].text
                        history.append({"role": "assistant", "content": annotation})

                if self.get_score_from_judge(annotation) != -1:
                    task['judge_score_1st_turn'] = self.get_score_from_judge(annotation)
                else:
                    task['judge_score_1st_turn'] = 5.0
                    print(f"No score found in the response, please inspect and retry.")
        
        if not os.path.exists(task['gen_2nd_turn']):
            task['judge_response_2nd_turn'] = 'Image not found, treated as a neutral entry.'
            task['judge_score_2nd_turn'] = 5.0
            print(f"Image 2 not found, treated as a neutral entry.")
            return task
        else:
            image2 = self.encode_image(task['gen_2nd_turn'])
            # second turn
            inputs = (image1, prompt2, image2)
            completion = self.GPT_decode(inputs, 'turn_2')
            if completion == 'Error':
                print(f"Error in GPT_decode, the entry treated as bad entry.")
                task['judge_response_2nd_turn'] = '[[0.0]]'
            elif completion == 'Blocked':
                print(f"{task}: \n\nBlocked, the entry treated as bad entry.")
                task['judge_response_2nd_turn'] = '[[0.0]]'
            else:
                annotation = completion.content[0].text
                task['judge_response_2nd_turn'] = annotation
                for i in range(self.FORMAT_MAXRETRY):
                    if self.get_score_from_judge(annotation) != -1:
                        task['judge_score_2nd_turn'] = self.get_score_from_judge(annotation)
                        break
                    else:
                        print(f"No score found in the response, retrying...")
                        history = text2image_claude_judge_turn2(*inputs)
                        history.append({"role": "assistant", "content": annotation})
                        completion = self.GPT_decode({"role": "user", "content": "Continue your judgment and finish by outputting a final rating with the above-mentioned format, i.e., the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Your rating: "}, 'append_message', history)
                        annotation = completion.content[0].text
                        history.append({"role": "assistant", "content": annotation})

                if self.get_score_from_judge(annotation) != -1:
                    task['judge_score_2nd_turn'] = self.get_score_from_judge(annotation)
                else:
                    task['judge_score_2nd_turn'] = 5.0
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
    

########################Gemini########################
class GeminiJudgeText2Image:
    def __init__(self, args):
        self.args = args
        self.JUDGE = args.judge_model
        self.FIX_INTERVAL_SECOND = 0
        self.MAX_RETRY_NUM = 999999
        self.MAX_NEW_TOKENS = 999
        
        self.FORMAT_MAXRETRY = 10

        load_dotenv()
        genai.configure(api_key=os.getenv('k_g'))
        self.model = genai.GenerativeModel(self.JUDGE)
        
        self.safety_settings={
            'harm_category_harassment':'block_none',
            'harm_category_hate_speech': 'block_none',
            'harm_category_sexually_explicit': 'block_none',
            'harm_category_dangerous_content': 'block_none'
            }
        
    @staticmethod
    def encode_image(image_path):
        return PIL.Image.open(image_path)
        
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
            formated = text2image_gemini_judge_turn1(*inputs)
        elif mode == 'turn_2':
            formated = text2image_gemini_judge_turn2(*inputs)
        elif mode == 'append_message' and history:
            history.append(inputs)
            formated = history
        else:
            raise ValueError(f"Invalid mode: {mode}.")
        
        return formated
    
    def _GPT_decode(self, inputs, mode, history=[]):
        completion = self.model.generate_content(
                            self.format_prompts(inputs, mode, history),
                            generation_config=genai.types.GenerationConfig(
                                candidate_count=1,
                                max_output_tokens=self.MAX_NEW_TOKENS,
                                ),
                            safety_settings=self.safety_settings,
                        )
        time.sleep(self.FIX_INTERVAL_SECOND)
        return completion.text


    def GPT_decode(self, inputs, mode, history=[]):
        delay = 1
        blocked = 0
        for i in range(self.MAX_RETRY_NUM):
            try:
                response_content = self._GPT_decode(inputs, mode, history)
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
        prompt1 = task['first_turn_user_prompt']
        prompt2 = task['second_turn_user_prompt']
        
        if not os.path.exists(task['gen_1st_turn']):
            task['judge_response_1st_turn'] = 'Image not found, treated as a neutral entry.'
            task['judge_score_1st_turn'] = 5.0
            task['judge_response_2nd_turn'] = 'Image not found, treated as a neutral entry.'
            task['judge_score_2nd_turn'] = 5.0
            print(f"Image 1 not found, treated as a neutral entry.")
            return task
        else:
            image1 = self.encode_image(task['gen_1st_turn'])
            
            # first turn
            inputs = (prompt1, image1)
            
            completion = self.GPT_decode(inputs, 'turn_1')
            if completion == 'Error':
                print(f"Error in GPT_decode, the entry treated as bad entry.")
                task['judge_response_1st_turn'] = '[[0.0]]'
            elif completion == 'Blocked':
                print(f"{task}: \n\nBlocked, the entry treated as bad entry.")
                task['judge_response_1st_turn'] = '[[0.0]]'
            else:
                annotation = completion
                task['judge_response_1st_turn'] = annotation
                for i in range(self.FORMAT_MAXRETRY):
                    if self.get_score_from_judge(annotation) != -1:
                        task['judge_score_1st_turn'] = self.get_score_from_judge(annotation)
                        break
                    else:
                        print(f"No score found in the response, retrying...")
                        history = text2image_gemini_judge_turn1(*inputs)
                        history.append(annotation)
                        completion = self.GPT_decode("Continue your judgment and finish by outputting a final rating with the above-mentioned format, i.e., the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Your rating: ", 'append_message', history)
                        annotation = completion
                        history.append(annotation)

                if self.get_score_from_judge(annotation) != -1:
                    task['judge_score_1st_turn'] = self.get_score_from_judge(annotation)
                else:
                    task['judge_score_1st_turn'] = 5.0
                    print(f"No score found in the response, please inspect and retry.")
        
        if not os.path.exists(task['gen_2nd_turn']):
            task['judge_response_2nd_turn'] = 'Image not found, treated as a neutral entry.'
            task['judge_score_2nd_turn'] = 5.0
            print(f"Image 2 not found, treated as a neutral entry.")
            return task
        else:
            image2 = self.encode_image(task['gen_2nd_turn'])
            # second turn
            inputs = (image1, prompt2, image2)
            completion = self.GPT_decode(inputs, 'turn_2')
            if completion == 'Error':
                print(f"Error in GPT_decode, the entry treated as bad entry.")
                task['judge_response_2nd_turn'] = '[[0.0]]'
            elif completion == 'Blocked':
                print(f"{task}: \n\nBlocked, the entry treated as bad entry.")
                task['judge_response_2nd_turn'] = '[[0.0]]'
            else:
                annotation = completion
                task['judge_response_2nd_turn'] = annotation
                for i in range(self.FORMAT_MAXRETRY):
                    if self.get_score_from_judge(annotation) != -1:
                        task['judge_score_2nd_turn'] = self.get_score_from_judge(annotation)
                        break
                    else:
                        print(f"No score found in the response, retrying...")
                        history = text2image_gemini_judge_turn2(*inputs)
                        history.append(annotation)
                        completion = self.GPT_decode("Continue your judgment and finish by outputting a final rating with the above-mentioned format, i.e., the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\". Your rating: ",'append_message', history)
                        annotation = completion
                        history.append(annotation)

                if self.get_score_from_judge(annotation) != -1:
                    task['judge_score_2nd_turn'] = self.get_score_from_judge(annotation)
                else:
                    task['judge_score_2nd_turn'] = 5.0
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