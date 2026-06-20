import numpy as np
import re
import ast
from llm_sdk import Small_LLM_Model

def test_model(manager, input):
    print("Loading model... (This might take a few seconds to load into memory)")
    # Initializes the model. It automatically handles the device (cpu/mps/cuda)
    model = Small_LLM_Model()
    results = dict()
    
    functions = list()
    for fn in manager.definition_functions:
        functions.append(fn['name'])
    
    prompt = f"""the function names: {functions},
input: {input},
the Answr: the best function name and parameters of it are """

    print(f"\nOriginal Prompt: '{input}'\n")
    
    # 1. Encode the text into a 2D tensor
    tensor_ids = model.encode(prompt)
    
    # 2. Convert the 2D tensor into a flat Python list of integers 
    # tensor_ids[0] grabs the inner array, .tolist() makes it a standard Python list
    input_ids = tensor_ids[0].tolist() 
    
    max_tokens_to_generate = 10
    
    for i in range(max_tokens_to_generate):
        # 3. Get the logits (scores) for the next token based on our list of IDs
        logits = model.get_logits_from_input_ids(input_ids)
        
        # 4. Use numpy to find the index of the highest score
        next_token_id = int(np.argmax(logits))
        
        # 5. Append the new token ID to our sequence
        input_ids.append(next_token_id)
        
        # 6. Decode the sequence to see what it generated so far
        current_text = model.decode(input_ids)
        # print(f"Step {i+1}: {current_text}")

    print("\n--- Final Generation ---")
    # print(model.decode(input_ids))
    response = model.decode(input_ids)
    print(response)

    source = {
        "fn_add_numbers": ["a", "b"],
        "fn_greet": ["name"],
        "fn_reverse_string": ["s"],
        "fn_get_square_root": ["a"],
        "fn_substitute_string_with_regex": ["source_string", "regex", "replacement"],
    }

    results.update({"prompt": input})

    func = ""
    data = response.split()
    for element in data[::-1]:
        if element.startswith("fn_"):
            func = element
            results.update({"name": element})
            break

    for element in data[::-1]:
        if element.startswith("["):
            parameters = ast.literal_eval(element)
            keys = source[func]
            for k, v in zip(keys, parameters):
                results["parameters"][k] = v
            break
    print(results)
    # match = re.search(r"fn_\d+", response)
    # print(match)
    # if not match:
    #     print("Output Error!")
    #     # exit(1)
    # else:
    #     fnc = match.group()
    #     print(fnc)
    

# if __name__ == "__main__":
#     test_model()