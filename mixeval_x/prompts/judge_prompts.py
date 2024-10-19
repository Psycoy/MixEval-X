############Judge Prompts for Close-ended Free-form Parser############
image2text_gpt_judge_for_closeended_freeform = lambda prompt, gold_ans, response: [
{"role": "system", "content": f"In this task, I want you to act as a judge."},
{"role": "user", "content": f'''You will be provided with a question, its golden answer(s), and the model's answer, while the context of the question, which is one or more images, is not given here. Your task is to judge how correct the model's answer is based on the golden answer(s), without seeing the input images of the question, and then give a correctness score. The correctness score should be one of the below numbers: 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Your should first briefly give your reasoning process regarding how the model's answer conforms to or contradicts the golden answer(s), and then give the correctness score. The correctness score must strictly follow this format: \"[[score]]\", e.g., \"The correctness score: [[0.5]]\". Below are some examples. 

Example 1: 
Question: what is this advertising?
Golden Answer(s): <answer 1> garden annual; <answer 2> seeds; <answer 3> seeds; <answer 4> seeds; <answer 5> seeds; <answer 6> seeds; <answer 7> seeds; <answer 8> seeds; <answer 9> seeds; <answer 10> cole's garden annual
Model's Answer: Seed
Your Judgment: The golden answers consistently mention "seeds" suggesting an advertisement for a seed catalog. The model's answer, "Seed", aligns exactly with this description. The Correctness Score: [[1.0]]

Example 2: 
Question: Who is making a face?
Golden Answer: <answer 1> child
Model's Answer: A man.
Your Judgment: The golden answer specifies a "child" making a face, but the model answered "A man", which is incorrect as it refers to a different age group. The Correctness Score: [[0.0]]

Example 3: 
Question: what road is to the right?
Golden Answer: <answer 1> troublesome valley rd; <answer 2> troublesome valley rd.; <answer 3> troublesome valley; <answer 4> troublesome valley road; <answer 5> valley road; <answer 6> troublesome valley; <answer 7> troublesome valley road; <answer 8> troublesome valley ; <answer 9> troublesome valley rd; <answer 10> troublesome valley rd.
Model's Answer: troublesome road
Your Judgment: The golden answers all specify the name of the road as "troublesome valley rd" or variations of this phrase with consistent reference to "troublesome valley." The model's answer, "troublesome road," captures the "troublesome" aspect but omits the critical "valley" part of the name, which is crucial for full accuracy. Thus, the model's answer partially matches the golden answer but lacks complete specificity. The Correctness Score: [[0.6]]

Note that each one of the golden answers is considered correct. Thus if the model's answer matches any one of the golden answers, it should be considered correct. Judge the below case, give the brief reasoning process and the correctness score.

Question: {prompt}
Golden Answer(s): {gold_ans}
Model's Answer: {response}
Your Judgment: 
'''}
]

video2text_gpt_judge_for_closeended_freeform = lambda prompt, gold_ans, response: [
{"role": "system", "content": f"In this task, I want you to act as a judge."},
{"role": "user", "content": f'''You will be provided with a question, its golden answer(s), and the model's answer, while the context of the question, which is one or more videos, is not given here. Your task is to judge how correct the model's answer is based on the golden answer(s), without seeing the input videos of the question, and then give a correctness score. The correctness score should be one of the below numbers: 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Your should first briefly give your reasoning process regarding how the model's answer conforms to or contradicts the golden answer(s), and then give the correctness score. The correctness score must strictly follow this format: \"[[score]]\", e.g., \"The correctness score: [[0.5]]\". Below are some examples. 

Example 1:
Question: what does this video want to express
Golden Answer(s): <answer 1> introduce method of playing
Model's Answer: Volleyball serve \n
Your Judgment: The model's answer "Volleyball serve" suggests a specific action, which may be part of what the video demonstrates. However, it misses the broader educational intent implied by the golden answer "introduce method of playing". Therefore, the answer is partially correct. The Correctness Score: [[0.5]]

Example 2:
Question: who do two other boys with surprised looks assist up?
Golden Answer(s): <answer 1> boy
Model's Answer: Boy.
Your Judgment: The model's answer "Boy." precisely matches the golden answer which states the two other boys assist a "boy". The Correctness Score: [[1.0]]

Example 3:
Question: what did the lady do at the end of the video after their performance
Golden Answer(s): <answer 1> picks up her phone
Model's Answer: Nothing.
Your Judgment: The model's answer "Nothing." directly contradicts the golden answer which states that the lady "picks up her phone" at the end of the video after their performance. Since the model's response completely misses the specific action described in the golden answer, it is incorrect. The Correctness Score: [[0.0]]

Note that each one of the golden answers is considered correct. Thus if the model's answer matches any one of the golden answers, it should be considered correct. Judge the below case, give the brief reasoning process and the correctness score.

Question: {prompt}
Golden Answer(s): {gold_ans}
Model's Answer: {response}
Your Judgment: 
'''}
]

audio2text_gpt_judge_for_closeended_freeform = lambda prompt, gold_ans, response: [
{"role": "system", "content": f"In this task, I want you to act as a judge."},
{"role": "user", "content": f'''You will be provided with a question, its golden answer(s), and the model's answer, while the context of the question, which is one or more audios, is not given here. Your task is to judge how correct the model's answer is based on the golden answer(s), without seeing the input audios of the question, and then give a correctness score. The correctness score should be one of the below numbers: 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Your should first briefly give your reasoning process regarding how the model's answer conforms to or contradicts the golden answer(s), and then give the correctness score. The correctness score must strictly follow this format: \"[[score]]\", e.g., \"The correctness score: [[0.5]]\". Below are some examples. 

Example 1:
Question: Are the people isolated?
Golden Answer(s): <answer 1> no
Model's Answer: yes
Your Judgment: The model's answer contradicts the golden answer directly. The question asked if the people are isolated, to which the golden answer is "no," indicating that the people are not isolated. However, the model's answer is "yes," implying that the people are isolated. The correctness score: [[0.0]]

Example 2:
Question: Who is speaking?
Golden Answer(s): <answer 1> man
Model's Answer: men
Your Judgment: The model's answer is almost correct but imprecise. The question asked about the identity of the speaker, to which the golden answer specifies a singular "man." However, the model's answer is "men," which suggests multiple individuals rather than one. This small pluralization error suggests a  misunderstanding of the query about the exact number of people speaking. The correctness score: [[0.6]]

Example 3:
Question: What did you hear after the door slamming?
Golden Answer(s): <answer 1> dog making noise
Model's Answer: dog
Your Judgment: The model's answer "dog" matches the golden answer's essential element, "dog making noise," by correctly identifying the dog. Although it omits "making noise," it captures the key information needed. The correctness score: [[1.0]]

Note that each one of the golden answers is considered correct. Thus if the model's answer matches any one of the golden answers, it should be considered correct. Judge the below case, give the brief reasoning process and the correctness score.

Question: {prompt}
Golden Answer(s): {gold_ans}
Model's Answer: {response}
Your Judgment: 
'''}
]

############Judge Prompts for Close-ended Multiple-choice Parser############
image2text_gpt_judge_for_closeended_multiplechoice = lambda prompt, options, response: [
{"role": "system", "content": f"In this task, I want you to act as an option extractor."},
{"role": "user", "content": f'''You will be provided with a multiple-choice question, its options, and the model's answer, while the context of the question, which is one or more images, is not given here. Your task is to extract or judge which option is chosen by the model based on its response, without seeing the context of the question. The extracted option should be one of the provided option letters. Your should first briefly give your reasoning process, and then give the extracted option letter. The extracted option must strictly follow this format: \"[[option letter]]\", e.g., \"The option chosen by the model: [[A]]\".
Below are some examples. 

Example 1: 
Question: Where are the cast of the television show located in the image?
Options:
A. In the foreground
B. In the background
C. In the center
D. At the edges
Model's Answer: C. In the center
Your Judgment: The model's answer clearly states "C. In the center", indicating that the correct option, according to the model, is in the center. The option chosen by the model: [[C]].

Example 2: 
Question: <image_1> on the left was painted during the 
Options:
A. first or second century C. E.
B. sixth or seventh century C. E.
C. tenth or eleventh century C.E.
D. fourteenth or fifteenth century C. E.
Model's Answer: The correct answer is option D, the fourteenth or fifteenth century C.E.
Your Judgment: The model's response specifies "option D, the fourteenth or fifteenth century C.E." directly as the correct answer. The option chosen by the model: [[D]].   

Example 3: 
Question: what does the diagram show's you information about
Options:
A. Photosynthesis
B. The plant getting fed
C. A picture of the plant
D. What happens to a plant daily
Model's Answer: The diagram shows the process of photosynthesis, which is the process by which plants convert sunlight, carbon dioxide, and water into oxygen and glucose. 
Your Judgment: The model's answer mentions "the process of photosynthesis," which directly corresponds to option A, "Photosynthesis". Therefore, the correct option according to the model is photosynthesis.  The option chosen by the model: [[A]].

Give the brief reasoning process and the extracted option for the below case:

Question: {prompt}
Options: 
{options}
Model's Answer: {response}
Your Judgment: 
'''}
]

video2text_gpt_judge_for_closeended_multiplechoice = lambda prompt, options, response: [
{"role": "system", "content": f"In this task, I want you to act as an option extractor."},
{"role": "user", "content": f'''You will be provided with a multiple-choice question, its options, and the model's answer, while the context of the question, which is one or more videos, is not given here. Your task is to extract or judge which option is chosen by the model based on its response, without seeing the context of the question. The extracted option should be one of the provided option letters. Your should first briefly give your reasoning process, and then give the extracted option letter. The extracted option must strictly follow this format: \"[[option letter]]\", e.g., \"The option chosen by the model: [[A]]\".
Below are some examples. 

Example 1:
Question: What did he do to the car?
Options:
A. Paint the car
B. Put plastic over the car
C. Put metal over the car
D. Cut the car
Model's Answer: put plastic over the car.
Your Judgment: The model's response directly aligns with option B, which is "Put plastic over the car." The response given is a paraphrase of this option without deviating in meaning. The option chosen by the model: [[B]]

Example 2:
Question: How did Eddie know Pam and Justin before Justin was killed?
Options:
A. They were part of the theater company
B. They were high school friends
C. They went to college together
D. They were cousins
E. They were siblings
Model's Answer: A.
Your Judgment: The model's answer directly provides the option letter "A." The option chosen by the model: [[A]]

Example 3:
Question: why do the people move in the same manner
Options:
A. uniform
B. dancing with the baby
C. exercising together
D. stay together
E. singing and dancing
Model's Answer: sing and dance
Your Judgment: The model's response "sing and dance" closely aligns with option E, which is "singing and dancing." The response provided is a direct paraphrase of this option, modifying only slightly the form of the words (from gerund to infinitive) but maintaining the same core activities described in the option. The option chosen by the model: [[E]]

When you think that the model's answer does not match any of the given options, please choose the option that is the closest to the model's answer.
Give the brief reasoning process and the extracted option for the below case. 

Question: {prompt}
Options: 
{options}
Model's Answer: {response}
Your Judgment: 
'''}
]


text2image_gpt_judge_turn1 = lambda prompt1, image1: [
{"role": "system", "content": f"In this task, you will act as an impartial judge for image generation tasks."},
{"role": "user", "content": [{'type': 'text', 'text': f'''
Please act as an impartial judge and evaluate the quality of an image generated by an AI assistant given the provided user prompt or caption. 

You must first analyze the generated image based on the provided prompt carefully. After providing your analysis, you must give the final score on a scale of 1 to 10 (1 means extremely bad and 10 means extremely good), and the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n

You should consider the following core aspects when analyzing the image:
1. **Alignment**: Assess how accurately the image reflects the given prompt. Check if all elements and requirements are correctly represented.
2. **Realism**: Judge if the image looks realistic and natural.
3. **Quality**: Identify if there's any flaw in the image, such as distortion, blurriness, or illogical representation of facial features, limbs, fingers, objects, or text. In addition, evaluate the overall quality of the image.

Analyze and judge the below case:

Generation Prompt/Caption: {prompt1}
Generated Image:
'''
},
{
"type": "image_url",
"image_url": {
    "url": f"data:image/jpg;base64,{image1}"
}
},
{
"type": "text",
"text": f'''
Your Analysis and Judgment: 
'''
}
                             ]}
]

text2image_gpt_judge_turn2 = lambda image1, prompt1, image2: [
{"role": "system", "content": f"In this task, you will act as an impartial judge for an image editing task."},
{"role": "user", "content": [{'type': 'text', 'text': f'''
You will be provided with an image to edit, the user prompt to edit the image, and the edited image. Your task is to evaluate the quality of the edited image based on the given information.

You must first analyze the edited image based on the provided editing prompt and the image to edit carefully. After providing your analysis, you must give the final score on a scale of 1 to 10 (1 means extremely bad and 10 means extremely good), and the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n

You should consider the following core aspects when analyzing the image:
1. **Alignment**: Assess how accurately the edited image reflects the changes indicated in the given editing prompt. Check if all elements and requirements are correctly meeted.
2. **Consistency**: Evaluate if the edited image is consistent with the original image in terms of details, style, color, overall appearance, etc.
3. **Realism**: Judge if the edited image looks realistic and natural after the editing process.
4. **Quality**: Identify if there's any flaw in the edited image, such as distortion, blurriness, or illogical representation of facial features, limbs, fingers, objects, or text. In addition, evaluate the overall quality of the image.


Analyze and judge the below case:

Editing Prompt/Caption: {prompt1}
The Image to Edit:
'''},
{
"type": "image_url",
"image_url": {
    "url": f"data:image/jpg;base64,{image1}"
}
},
{
"type": "text",
"text": f'''
The Edited Image: 
'''
},
{
"type": "image_url",
"image_url": {
    "url": f"data:image/jpg;base64,{image2}"
}
},
{
"type": "text",
"text": f'''
Your Analysis and Judgment: 
'''
}
                             ]}
]

text2image_claude_judge_turn1 = lambda prompt1, image1: [
{"role": "user", "content": [{'type': 'text', 'text': f'''In this task, you will act as an impartial judge for image generation tasks.
                              
Please act as an impartial judge and evaluate the quality of an image generated by an AI assistant given the provided user prompt or caption. 

You must first analyze the generated image based on the provided prompt carefully. After providing your analysis, you must give the final score on a scale of 1 to 10 (1 means extremely bad and 10 means extremely good), and the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n

You should consider the following core aspects when analyzing the image:
1. **Alignment**: Assess how accurately the image reflects the given prompt. Check if all elements and requirements are correctly represented.
2. **Realism**: Judge if the image looks realistic and natural.
3. **Quality**: Identify if there's any flaw in the image, such as distortion, blurriness, or illogical representation of facial features, limbs, fingers, objects, or text. In addition, evaluate the overall quality of the image.

Analyze and judge the below case:

Generation Prompt/Caption: {prompt1}
Generated Image:
'''
},
{
"type": "image",
"source": {
    "type": "base64",
    "media_type": "image/jpeg",
    "data": image1
}
},
{
"type": "text",
"text": f'''
Your Analysis and Judgment: 
'''
}
                             ]}
]

text2image_claude_judge_turn2 = lambda image1, prompt1, image2: [
{"role": "user", "content": [{'type': 'text', 'text': f'''In this task, you will act as an impartial judge for an image editing task.
                              
You will be provided with an image to edit, the user prompt to edit the image, and the edited image. Your task is to evaluate the quality of the edited image based on the given information.

You must first analyze the edited image based on the provided editing prompt and the image to edit carefully. After providing your analysis, you must give the final score on a scale of 1 to 10 (1 means extremely bad and 10 means extremely good), and the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n

You should consider the following core aspects when analyzing the image:
1. **Alignment**: Assess how accurately the edited image reflects the changes indicated in the given editing prompt. Check if all elements and requirements are correctly meeted.
2. **Consistency**: Evaluate if the edited image is consistent with the original image in terms of details, style, color, overall appearance, etc.
3. **Realism**: Judge if the edited image looks realistic and natural after the editing process.
4. **Quality**: Identify if there's any flaw in the edited image, such as distortion, blurriness, or illogical representation of facial features, limbs, fingers, objects, or text. In addition, evaluate the overall quality of the image.


Analyze and judge the below case:

Editing Prompt/Caption: {prompt1}
The Image to Edit:
'''},
{
"type": "image",
"source": {
    "type": "base64",
    "media_type": "image/jpeg",
    "data": image1
}
},
{
"type": "text",
"text": f'''
The Edited Image: 
'''
},
{
"type": "image",
"source": {
    "type": "base64",
    "media_type": "image/jpeg",
    "data": image2
}
},
{
"type": "text",
"text": f'''
Your Analysis and Judgment: 
'''
}
                             ]}
]


text2image_gemini_judge_turn1 = lambda prompt1, image1: [
f'''In this task, you will act as an impartial judge for image generation tasks.
                              
Please act as an impartial judge and evaluate the quality of an image generated by an AI assistant given the provided user prompt or caption. 

You must first analyze the generated image based on the provided prompt carefully. After providing your analysis, you must give the final score on a scale of 1 to 10 (1 means extremely bad and 10 means extremely good), and the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n

You should consider the following core aspects when analyzing the image:
1. **Alignment**: Assess how accurately the image reflects the given prompt. Check if all elements and requirements are correctly represented.
2. **Realism**: Judge if the image looks realistic and natural.
3. **Quality**: Identify if there's any flaw in the image, such as distortion, blurriness, or illogical representation of facial features, limbs, fingers, objects, or text. In addition, evaluate the overall quality of the image.

Analyze and judge the below case:

Generation Prompt/Caption: {prompt1}
Generated Image:
''',
image1,
f'''
Your Analysis and Judgment: 
'''
]

text2image_gemini_judge_turn2 = lambda image1, prompt1, image2: [
f'''In this task, you will act as an impartial judge for an image editing task.
                              
You will be provided with an image to edit, the user prompt to edit the image, and the edited image. Your task is to evaluate the quality of the edited image based on the given information.

You must first analyze the edited image based on the provided editing prompt and the image to edit carefully. After providing your analysis, you must give the final score on a scale of 1 to 10 (1 means extremely bad and 10 means extremely good), and the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n

You should consider the following core aspects when analyzing the image:
1. **Alignment**: Assess how accurately the edited image reflects the changes indicated in the given editing prompt. Check if all elements and requirements are correctly meeted.
2. **Consistency**: Evaluate if the edited image is consistent with the original image in terms of details, style, color, overall appearance, etc.
3. **Realism**: Judge if the edited image looks realistic and natural after the editing process.
4. **Quality**: Identify if there's any flaw in the edited image, such as distortion, blurriness, or illogical representation of facial features, limbs, fingers, objects, or text. In addition, evaluate the overall quality of the image.


Analyze and judge the below case:

Editing Prompt/Caption: {prompt1}
The Image to Edit:
''',
image1,
f'''
The Edited Image: 
''',
image2,
f'''
Your Analysis and Judgment: 
'''
]

text2action_gpt_judge = lambda task_description, allowed_actions, visible_objects, already_executed_steps, target, model_response : [
    {"role": "system", "content": f"In this task, you will act as an impartial judge for a real-world planning task."},
    {"role": "user", "content": f'''Your job is to evaluate the quality of the action-object sequences planned by an AI assistant for a real-world task. You will be provided with the Task Description, Allowed Actions, Visible Objects, Already Executed Action-Object Sequences, the target, and the model's response. The 'Task Description' is a user instruction that instructs the AI assistant, which is being evaluated, to complete the task. The 'Allowed Actions' is a list of actions that are allowed to be used by the AI assistant to complete the task. The 'Visible Objects' is a list of objects that are assumed to be visible to the AI assistant when it's completing the task. The 'Already Executed Action-Object Sequences' is a list of action-object sequences that are assumed to have been completed by the AI assistant at the moment of starting the planning. The 'Reference Answer' is an example action-object sequence output for your reference, which is annotated by a human and may not be the only correct answer. The 'Model Response' is the output of the AI assistant you are evaluating.

Your task is to analyze the model's response and evaluate how well it plans given the above-mentioned information and the reference answer. After providing your analysis, you must give the final score on a scale of 1 to 10 (1 means extremely bad and 10 means extremely good), and the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n

Below is a simplified example of how to judge the model's response:

**Start of Example**
Task Description: Put a heated egg in the sink.
Allowed Actions: [OpenObject], [CloseObject], [PickupObject], [PutObject], [ToggleObjectOn], [ToggleObjectOff], [SliceObject], [Navigation]
Visible Objects: <microwave>, <sink>, <toaster>, <coffee maker>, <fridge>, <blender>, <potato>, <bows>, <egg>, <garbagecan>
Already Executed Action-Object Sequences: [Navigation] <fridge>, [OpenObject] <fridge>, [PickupObject] <egg>, [CloseObject] <fridge>, [Navigation] <microwave>, [PutObject] <egg> <microwave>
Reference Answer: [ToggleObjectOn] <microwave>, [ToggleObjectOff] <microwave>, [PickupObject] <egg>, [Navigation] <sink>, [PutObject] <egg> <sink>
Model Response: [PickupObject] <egg>, [Navigation] <sink>, [PutObject] <egg> <sink>
Your Analysis and Judgment: The model's response omits crucial steps for heating the egg, assuming it is already heated without evidence from prior actions. It correctly performs the transport and placement of the egg, using appropriate actions and objects. However, by neglecting the heating process essential to the task description, the response is incomplete. My Final Rating: [[3]].
**End of Example**

With the above description and example, analyze and judge the below case:

Task Description: {task_description}
Allowed Actions: {allowed_actions}
Visible Objects: {visible_objects}
Already Executed Action-Object Sequences: {already_executed_steps}
Reference Answer: {target}
Model Response: {model_response}
Your Analysis and Judgment:
'''}
]


image2action_gpt_judge = lambda example_image, task_description, allowed_actions, image, already_executed_steps, target, model_response : [
    {"role": "system", "content": f"In this task, you will act as an impartial judge for a real-world planning task."},
    {"role": "user", "content": [{'type': 'text', 'text': f'''Your job is to evaluate the quality of the action-object sequences planned by an AI assistant with visual perception for a real-world task. You will be provided with the Task Description, Allowed Actions, Visible Objects, Already Executed Action-Object Sequences, the target, and the model's response. The 'Task Description' is a user instruction that instructs the AI assistant, which is being evaluated, to complete the task. The 'Allowed Actions' is a list of actions that are allowed to be used by the AI assistant to complete the task. The 'Visible Objects' is a list of objects that are assumed to be visible to the AI assistant when it's completing the task. Note that some invisible objects may still be usable to the AI assistant, but their existence must be consistent with the commonsense. The 'Already Executed Action-Object Sequences' is a list of action-object sequences that are assumed to have been completed by the AI assistant at the moment of starting the planning. The 'Reference Answer' is an example action-object sequence output for your reference, which is annotated by a human and may not be the only correct answer. The 'Model Response' is the output of the AI assistant you are evaluating.

Your task is to analyze the model's response and evaluate how well it plans given the above-mentioned information and the reference answer. After providing your analysis, you must give the final score on a scale of 1 to 10 (1 means extremely bad and 10 means extremely good), and the rating must strictly follow this format: \"[[rating]]\", for example: \"Rating: [[5]]\".

Below is a simplified example of how to judge the model's response:

**Start of Example**
Task Description: Get the egg from the fridge, and put the heated egg in the sink.
Allowed Actions: [OpenObject], [CloseObject], [PickupObject], [PutObject], [ToggleObjectOn], [ToggleObjectOff], [SliceObject], [Navigation]
Visible Objects: '''},
{
"type": "image_url",
"image_url": {
    "url": f"data:image/jpg;base64,{example_image}"
}
},
{'type': 'text', 'text': f'''
Already Executed Action-Object Sequences: [Navigation] <fridge>, [OpenObject] <fridge>, [PickupObject] <egg>, [CloseObject] <fridge>, [Navigation] <microwave>, [PutObject] <egg> <microwave>
Reference Answer: [ToggleObjectOn] <microwave>, [ToggleObjectOff] <microwave>, [PickupObject] <egg>, [Navigation] <sink>, [PutObject] <egg> <sink>
Model Response: [PickupObject] <egg>, [Navigation] <sink>, [PutObject] <egg> <sink>
Your Analysis and Judgment: The model's response omits crucial steps for heating the egg, assuming it is already heated without evidence from prior actions. It correctly performs the transport and placement of the egg, using appropriate actions and objects. However, by neglecting the heating process essential to the task description, the response is incomplete. My Final Rating: [[3]].
**End of Example**

With the above description and example, analyze and judge the below case:

Task Description: {task_description}
Allowed Actions: {allowed_actions}
Visible Objects: '''},
{
"type": "image_url",
"image_url": {
    "url": f"data:image/jpg;base64,{image}"
}
},
{'type': 'text', 'text': f'''
Already Executed Action-Object Sequences: {already_executed_steps}
Reference Answer: {target}
Model Response: {model_response}
Your Analysis and Judgment: '''}
]}
]