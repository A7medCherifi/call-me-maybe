import numpy as np
import re
import ast
from llm_sdk import Small_LLM_Model

def process_llm_output(raw_llm, prompt):
    # 1. Safely convert the string into a real Python list
    try:
        extracted_data = ast.literal_eval(raw_llm)
    except Exception as e:
        print(f"Error parsing model output: {e}")
        return None

    # 2. Build your final dictionary exactly how you want it
    final_dict = {
        "prompt": prompt,
        "name": extracted_data[0],          # The function name
        "parameters": extracted_data[1]     # The parameters dictionary
    }
    
    return final_dict


def test_model(manager, input):
    print("Loading model... (This might take a few seconds to load into memory)")
    # Initializes the model. It automatically handles the device (cpu/mps/cuda)
    model = Small_LLM_Model()
    results = dict()
    
    functions = ""
    for fn in manager.definition_functions:
        functions += f"Name: {fn['name']} | Parameters: {fn['parameters']}\n"

    prompt = f"""Output only valid JSON values.
    Rules:
        1. if a parameter type is number cast it to float.
    Functions:
        {functions}
    Example:
        {{
            "prompt": "Input",
            "name": "Function name",
            "parameters: "Parameters of the function"
        }}
    Input:
        {input}
    JSON:
    {{"prompt": {input}, "name": """

    print(f"\nOriginal Prompt: '{input}'\n")
    
    # 1. Encode the text into a 2D tensor
    tensor_ids = model.encode(prompt)
    
    # 2. Convert the 2D tensor into a flat Python list of integers 
    # tensor_ids[0] grabs the inner array, .tolist() makes it a standard Python list
    input_ids = tensor_ids[0].tolist() 
    
    max_tokens_to_generate = 70
    
    text = ""
    for i in range(max_tokens_to_generate):
        # 3. Get the logits (scores) for the next token based on our list of IDs
        logits = model.get_logits_from_input_ids(input_ids)
        
        # 4. Use numpy to find the index of the highest score
        next_token_id = int(np.argmax(logits))

        # 6. Decode the sequence to see what it generated so far
        current_text = model.decode([next_token_id])
        
        # 5. Append the new token ID to our sequence
        text += current_text
        input_ids.append(next_token_id)
        # print(f"Step {i+1}: {current_text}")
        print(current_text, end="", flush=True)
        if "]" in current_text:
            break
    # print(model.decode(input_ids))
    # response = model.decode(input_ids)
    # print(text)

    # result = process_llm_output(text, input)
    # print(result)
    # results.update({"prompt": input})
