import numpy as np
import re
import ast
import time
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


def test_model(manager, inputs):
    model = Small_LLM_Model()
    resource = dict()
    functions = ""
    all_prompts = list()

    start = time.perf_counter()
    inject_parameter_str = '"parameters": {'
    parameter_tensor = model.encode(inject_parameter_str)
    parameter_ids = parameter_tensor[0].tolist()

    forced_after_comma = ' '
    after_comma_tensor = model.encode(forced_after_comma)
    after_comma_id = after_comma_tensor[0].tolist()


    for fn in manager.definition_functions:
        functions += f"Name: {fn['name']} | Parameters: {fn['parameters']}\n"
        # keys = []
        # for key in fn['parameters'].keys():
        #   keys.append(key)
        resource[fn['name']] = [(name, value['type']) for name, value in fn['parameters'].items()]
        # print(resource[fn['name']])

    for element in manager.prompts_calling:
        all_prompts.append(element['prompt'])
    
    for input in all_prompts:
      prompt = f"""Extract the function name and parameters as a valid JSON object. \
      Functions: \
          {functions} \
      Example: \
          Input text: What is the sum of 2 and 3? \
          JSON output: \"name\": \"fn_add_numbers\", \"parameters\": {{"a": 2.0, "b": 3.0}}.\
      Rules: \
          1. If a parameter type is a Number cast it to a float. \
          2. Output ONLY the raw JSON. \
      Input Text: {input} \
          JSON output: \
  """
      json_start = "{" + f'"prompt": "{input}", "name": "'
      prompt += json_start
      tensor_ids = model.encode(prompt)
      input_ids = tensor_ids[0].tolist()
      text = json_start

      done_json = 0
      open_braces = 1
      closed_braces = 0
      func_name = ""
      parameters = ""

      i = 0
      next_arg = 1
      extract_func_name= True
      while True:
        logits = model.get_logits_from_input_ids(input_ids)
        next_token_id = int(np.argmax(logits))
    
        current_text = model.decode([next_token_id])

        if current_text.endswith(','):
          current_text += forced_after_comma
          input_ids.extend(after_comma_id)
        
        if not done_json:
          open_braces += current_text.count("{")
          closed_braces += current_text.count("}")

          injected = False
            
          if extract_func_name:
            if '"' not in current_text:
              func_name += current_text
            else:
              func_name += current_text.split('"')[0]
              extract_func_name = False
              text += current_text
            
              input_ids.extend(parameter_ids)
              text += inject_parameter_str
              open_braces += 1
              injected = True
          
          if 'parameters' in text and i < len(resource[func_name]):
            if next_arg:
              if resource[func_name][i][1] == 'string':
                par_str = f'"{resource[func_name][i][0]}": "'
              else:
                par_str = f'"{resource[func_name][i][0]}": '
              par_tensor = model.encode(par_str)
              par_ids = par_tensor[0].tolist()

              input_ids.extend(par_ids)
              text += par_str
              parameters += par_str
              i += 1
              next_arg = 0
              injected = True

            else:
              parameters += current_text
            
            if parameters.count(',') == i:
              next_arg = 1
      
          if not injected:
            input_ids.append(next_token_id)
            text += current_text
    
          if open_braces == closed_braces:
              if 'parameters' in text:
                text, braces, _ = text.rpartition('}}')
                text += braces
                done_json = 1

        else:
          break
        print(text)
      
      print("\n#################################################\n")

    elapsed = time.perf_counter() - start
    print(f"Time: {elapsed / 60:.2f}")

























