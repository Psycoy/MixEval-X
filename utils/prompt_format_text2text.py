import json

# Besides the prompt, the input to the model also requires the an input field. 
# for text2text, the input field is the text specified in 'context';
# for image2text, the input field is the image specified in 'image_id';
# for audio2text, the input field is the audio specified in 'audio_id';
# for video2text, the input field is the video specified in 'video_id'.


MULTI_CHOICE_PROMPT = "Answer with the option letter from the given choices directly."
FREE_FORM_PROMPT = "Answer the question using a single word or phrase."
FREE_FORM_PROMPT_QUAC = "Answer the question using a short excerpt (span) from the given text."
FREE_FORM_PROMPT_BBH = "Answer the question with a single word or phrase. When there are options contained in the question, answer with the option or option letter directly."

def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str

def construct_prompt_multichoice(entry):
    prompt = entry["prompt"]
    parsed_options = parse_options(entry["options"])
    prompt = f"{prompt}\n{parsed_options}\n{MULTI_CHOICE_PROMPT}"
    return prompt

def construct_prompt_freeform(entry):
    prompt = entry["prompt"]
    if entry["benchmark_name"] == "QuAc":
        prompt = f"{prompt}\n{FREE_FORM_PROMPT_QUAC}"
    elif entry["benchmark_name"] == "BBH":
        prompt = f"{prompt}\n{FREE_FORM_PROMPT_BBH}"
    else:
        prompt = f"{prompt}\n{FREE_FORM_PROMPT}"
    return prompt


if __name__ == "__main__":
    
    task_file_path = "/data/personal/nus-njj/InstructioninWild/cache/Datasets/WildBench/text2text/close_ended/query_data_ff/tasks.json"
    with open(task_file_path, "r") as f:
        data = json.load(f)
    for id, entry in data.items():
        if "options" in entry:
            prompt = construct_prompt_multichoice(entry)
        else:
            prompt = construct_prompt_freeform(entry)
        
        print(prompt)
