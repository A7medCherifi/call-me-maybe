import numpy as np
import re
import ast
from llm_sdk import Small_LLM_Model

def process_llm_output(raw_llm, prompt):
    try:
        extracted_data = ast.literal_eval(raw_llm)
    except Exception as e:
        print(f"Error parsing model output: {e}")
    return None
    final_dict = {
        "prompt": prompt,
        "name": extracted_data[0],
        "parameters": extracted_data[1]
    }
    return final_dict


def test_model(manager, input):
    model = Small_LLM_Model()
    results = dict()
    functions = ""
    all_prompts = list()
    for fn in manager.definition_functions:
        functions += f"Name: {fn['name']} | Parameters: {fn['parameters']}\n"
    for element in manager.prompts_calling:
        all_prompts.append(element['prompt'])
    
    for element in all_prompts:
        prompt = f"""Extract the function name and parameters from the Input Text and return them as a valid JSON object. \
        Functions available: \
            {functions} \
        
        Example: \
            Input text: What is the sum of 2 and 3? \
            JSON output: {{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {{"a": 2.0, "b": 3.0}}}}.\
        Rules: \
            1. If a parameter type is a Number, cast it to a float. \
            2. Output ONLY the raw JSON, Do not include conversational filler text. \
        
        Input Text: {element} \
            JSON output: \
"""
        json_start = "{" + f'"prompt": "{element}", "name": "'
        prompt += json_start
        tensor_ids = model.encode(prompt)
        input_ids = tensor_ids[0].tolist()
        text = json_start
        done_json = 0
        open_braces = 1
        closed_braces = 0
        
        while True:
            logits = model.get_logits_from_input_ids(input_ids)
            next_token_id = int(np.argmax(logits))
            
            current_text = model.decode([next_token_id])
            
            if not done_json:
                input_ids.append(next_token_id)
                text += current_text
                open_braces += current_text.count("{")
                closed_braces += current_text.count("}")
                if text.count('"') == 8:
                    inject_parameter_str = ' "parameters": {"'
                    parameter_tensor = model.encode(inject_parameter_str)
                    parameter_ids = parameter_tensor[0].tolist()
                    
                    input_ids.extend(parameter_ids)
                    text += inject_parameter_str
                    open_braces += 1
            
                if open_braces == closed_braces:
                    done_json = 1
            else:
                text = text.strip()
            if 'parameters' in text:
                break
            else:
                print("Invalid Input!")
                break
            print(text)
        
        print("#################################################")



