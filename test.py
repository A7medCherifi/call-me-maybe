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
    for fn in manager.definition_functions:
        functions += f"Name: {fn['name']} | Parameters: {fn['parameters']}\n"

    prompt = f"""You are a function calling program. Your task is to extract the function name and parameters from the input text and return them as a valid JSON object. \
    Functions available: \
    {functions} \

    Example: \
    Input text: What is the sum of 2 and 3? \
    JSON output: {{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {{"a": 2.0, "b": 3.0}}}} <EOS>\

    Rules: \
    1. If a parameter type is a Number, cast it to a float. \
    2. Output ONLY the raw JSON object with <EOS> AT THE END OF JSON and ALWAYS include it at the end of JSON. Do not include conversational filler text. \

    Input text: {input} \
    JSON output: \
  """

    # prompt = f"""You are a function calling program, where you get the function name and parameters from the input text as a JSON. \
    # Functions: {functions} \
    # Example JSON: Input test: What is the sum of 2 and 3?, JSON output: \{\"prompt\": \"What is the sum of 2 and 3?\", \"name\": \"fn_add_numbers\", \"parameters\": \{\"a\": 2.0, \"b\": 3.0\}\} <EOS> \
    # Rules: \
    #   1. if parameter type is Number make it float. Respond ONLY with valid JSON. \
    #   2. if you closed JSON with \"\}\}\" ADD this always into the end of it: \"<|endoftext|>\" \
    # Task: Input text: {input_text} \
    # JSON output: """

  # Output only valid JSON values.
  # Rules:
  #       1. if a parameter type is number cast it to float.
  #   Functions:
  #       {functions}
  #   Example:
  #       {{
  #           "prompt": "Input",
  #           "name": "Function name",
  #           "parameters: "Parameters of the function"
  #       }}
  #   Input:
  #       {input}
  #   JSON:
  # "prompt": {input}, "name":

    print(f"\nOriginal Prompt: '{input}'\n")

    json_start = "{"
    json_start += f'"prompt": {input}, "name":'
    prompt += json_start
    tensor_ids = model.encode(prompt)
    input_ids = tensor_ids[0].tolist()
    
    text = json_start
    endoftext = " > "
    done_json = 0

    open_braces = 1
    closed_braces = 0
    print(json_start, end="", flush=True)

    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = int(np.argmax(logits))

        current_text = model.decode([next_token_id])

        text += current_text
        input_ids.append(next_token_id)

        if not done_json:
          open_braces += current_text.count("{")
          closed_braces += current_text.count("}")

          if open_braces == closed_braces:
            done_json = 1
          print(current_text, end="", flush=True)
        
        else:
          if all(['prompt', 'name', 'parameters'], text):
            break
          endoftext += current_text
          if "<EOS>" in endoftext:
            break
          # if len(endoftext) > 7:
            # print(" < fuck.", end="", flush=True)
            # break
          print(f"\n{endoftext}")




    # print(model.decode(input_ids))
    # response = model.decode(input_ids)
    # print(text)

    # result = process_llm_output(text, input)
    # print(result)
    # results.update({"prompt": input})













