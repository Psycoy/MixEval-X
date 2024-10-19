# Model Output Formats

As illustrated in the README.md, you can run our provided command after preparing your model output files as specified in this document. For example, below is our provided grading command for the Image2Text benchmark:

```
python -m mixeval_x.compute_metrics_mmu \
    --benchmark mixeval_x_image2text_close \
    --model_response_dir THE_PATH_TO_MODEL_OUTPUT_FOLDER \
    --models_to_eval \
        gemini_1_5_pro \
        gemini_1_5_flash
```

Here, the `--model_response_dir` specifies the parent directory of your model directory. Your model directory contains the model output files. The model directory name is your model name that will be specified by `--models_to_eval`. Generally, your prepared model output files should have a structure like this:

```
    └── model_response_dir
        └── your_model_name
            │
            ├── file1.jsonl
            │
            └── file2.jsonl
            │
            └── ...
```

> The difference between model input (the benchmark data) and output (the model response file) is just the `"response"` field, i.e., each entry in your output file should keep all key-value pairs (including the 'id') of the input entry, with an additional `"response"` field representing the model's output.

🚨 **We show example model output structures and files in `mixeval_x/examples/`**.

The exact output structures and data formats are detailed below. 

<br><br><br>

## MMU Tasks (Image2Text, Video2Text, and Audio2Text)

### Structure
```
    └── model_response_dir
        └── your_model_name
            │
            ├── your_model_name_ff.jsonl
            │
            └── your_model_name_mp.jsonl
```
> Note that the Audio2Text benchmark doesn't have the multiple-choice subset, therefore the "`_mp`" file is not required.

### Output Data Format

Free-form (your_model_name_ff):
```
[
    {
        "id": "1", 
        "problem_type": "free-form", 
        "image_id": "1.jpg", 
        "prompt": "Where are the people that are standing looking at?", 
        "target": ["away"], 
        "benchmark_name": "GQA", 
        "response": "The kite"
    },
    ...
]
```

Multiple-choice (your_model_name_mp):
```
[
    {
        "problem_type": "single-choice", 
        "image_id": "1.jpg", 
        "prompt": "Is the man's face clearly visible in the image?", 
        "options": ["Yes", "No"], 
        "target": [1], 
        "benchmark_name": "Q-Bench", 
        "response": "B."
    },
    ...
]
```

## Agent Tasks

### Text2Action

Structure:
```
    └── model_response_dir
        └── your_model_name
            │
            └── your_model_name_t2a.jsonl
```

Output Data Format:
```
[
    {
        "id": "1", 
        "task description": "Take a photo of the Oso Nabukete cave entrance.", 
        "allowed actions": "[Navigation], [Photograph], [InteractWithObject], [PickupObject], [PutObject], [UseObject], [ChargeDevice], [CheckBattery], [AdjustSettings], [Wait], [Speak], [Listen]", 
        "visible objects": "<camera>, <smartphone>, <cave entrance>, <tourists>, <guide>, <charging station>, <backpack>, <water bottle>, <map>, <brochure>, <battery>, <tripod>", "already executed steps": "[Navigation] <cave entrance>, [PickupObject] <camera>, [CheckBattery] <camera>", 
        "target": "[AdjustSettings] <camera>, [Photograph] <cave entrance>", 
        "response": "[AdjustSettings] <camera>, [ChargeDevice] <camera> <charging station>, [PickupObject] <smartphone>, [UseObject] <smartphone> <camera>, [Photograph] <camera> <cave entrance>"
    },
    ...
]
```

### Image2Action
Structure:
```
    └── model_response_dir
        └── your_model_name
            │
            └── your_model_name_i2a.jsonl
```

Output Data Format:
```
[
    {
        "id": "1",
        "image_id": "7.jpg", 
        "task description": "Paint an object with blue color to resemble a blueberry.", 
        "allowed actions": "[PickupObject], [PutObject], [Navigation], [OpenContainer], [CloseContainer], [SelectPaintTool], [MixColor], [ApplyPaint], [CleanBrush], [InspectObject], [DryObject]", 
        "already executed steps": "[Navigation] <paint station>, [SelectPaintTool] <brush>, [MixColor] <blue paint>, [ApplyPaint] <object>", 
        "target": "[InspectObject] <object>, [DryObject] <object>", 
        "response": "[InspectObject] <object>, [DryObject] <object>, [CleanBrush] <brush>"
    },
    ...
]
```

## MMG Tasks (only supports Text2Image)

### Text2Image
Structure:
```
    └── model_response_dir
        └── your_model_name
            │
            └── your_model_name_t2i.jsonl
```

Output Data Format:
```
[
    {
        "id": "1",
        "first_turn_user_prompt": "Design a vibrant cityscape banner featuring the iconic Sydney Opera House and Harbour Bridge under a dazzling sunset, with the City of Sydney\u2019s logo prominently displayed in the foreground, ensuring it catches the eye against the vivid backdrop.", 
        "first_turn_caption": "A vibrant cityscape banner featuring the iconic Sydney Opera House and Harbour Bridge under a dazzling sunset, with the City of Sydney\u2019s logo prominently displayed in the foreground, ensuring it catches the eye against the vivid backdrop.", 
        "second_turn_user_prompt": "<generated_image_1> Remove the City of Sydney\u2019s logo.", 
        "second_turn_caption": "A vibrant cityscape banner featuring the iconic Sydney Opera House and Harbour Bridge under a dazzling sunset, ensuring it catches the eye against the vivid backdrop.", 
        "gen_1st_turn": "THE DIR PATH/1.jpg", 
        "gen_2nd_turn": "THE DIR PATH/2.jpg", 
    },
    ...
]
```


