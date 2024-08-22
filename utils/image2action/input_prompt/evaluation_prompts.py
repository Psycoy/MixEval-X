def construct_prompt_image2action(entry):
    '''
    This function constructs the prompt for the image2action task.
    When formatting the input to the VLMs, you many need to separate this function into interleaved parts.
    The 'example_image' is a placeholder for the example_image.jpg.
    The 'task_image' is a placeholder for the entry["image_id"].
    '''
    
    
    prompt = f'''You are a real-world agent, and you will plan action-object sequences for the real-world tasks. You will be provided with 'Task Description', 'Allowed Actions', 'Visible Objects', and 'Already Executed Action-Object Sequences'. The 'Task Description' is a user instruction that instructs you to complete the task. The 'Allowed Actions' is a list of actions that are allowed to be used by you to complete the task. The 'visible objects' is an image indicating the objects usable to you when you are completing the task. Note that some invisible objects may still be usable to you, but their existence must be consistent with the commonsense. The 'Already Executed Action-Object Sequences' is a list of action-object sequences that are assumed to have been completed by you at the moment. You need to plan the remaining action-object sequences to complete the task.

Below is a simplified example:

**Start of Example**
Task Description: Get the egg from the fridge, and put the heated egg in the sink.
Allowed Actions: [OpenObject], [CloseObject], [PickupObject], [PutObject], [ToggleObjectOn], [ToggleObjectOff], [SliceObject], [Navigation]
Visible Objects: {example_image}
Already Executed Action-Object Sequences: [Navigation] <fridge>, [OpenObject] <fridge>, [PickupObject] <egg>, [CloseObject] <fridge>, [Navigation] <microwave>, [PutObject] <egg> <microwave>
Your Planning: [ToggleObjectOn] <microwave>, [ToggleObjectOff] <microwave>, [Navigation] <sink>, [PickupObject] <egg>, [PutObject] <egg> <sink>
**End of Example**

With the above description and example, plan the remaining action-object sequences for the below task:

Task Description: {entry["task description"]}
Allowed Actions: {entry["allowed actions"]}
Visible Objects: {task_image}
Already Executed Action-Object Sequences: {entry["already executed steps"]}
Your Planning: 
'''
    
    return prompt