import numpy as np
import re
import ast
import time
from llm_sdk import Small_LLM_Model


class Model():
  def __init__(self):
    self.func_name = ""
    self.extract_func_name = True
    self.func_vocab = list()
    self.model = Small_LLM_Model()
    self.func_trie = {}


  # def get_func_name(self, next_token_id):
  #   if next_token_id not in self.func_vocab:
      
    # current_text = self.model.decode([next_token_id])
    # queue = list()
    # for func in self.func_name:
    #   if func.startswith(current_text):
    #     queue.append(func)

    # if len(queue) == 0:
    #   # Fix this
    #   pass

    # if len(queue) == 1:
    #   self.func_name = queue[0]
    #   self.extract_func_name = False

    # else:
    #   if '"' not in current_text:
    #     self.func_name += current_text
    #   else:
    #     self.func_name += current_text.split('"')[0]
    #     self.extract_func_name = False


    # if '"' not in current_text:
    #   self.func_name += current_text
    # else:
    #   self.func_name += current_text.split('"')[0]
    #   self.extract_func_name = False



  def test_model(self, manager, inputs):
    resource = dict()
    functions = ""
    all_prompts = list()
    allowed_funcs = set()
    

    start = time.perf_counter()
    inject_parameter_str = '"parameters": {'
    parameter_tensor = self.model.encode(inject_parameter_str)
    parameter_ids = parameter_tensor[0].tolist()

    forced_after_comma = ' '
    after_comma_tensor = self.model.encode(forced_after_comma)
    after_comma_id = after_comma_tensor[0].tolist()


    for fn in manager.definition_functions:
        functions += f"Name: {fn['name']} | Parameters: {fn['parameters']}\n"
        # keys = []
        # for key in fn['parameters'].keys():
        #   keys.append(key)
        resource[fn['name']] = [(name, value['type']) for name, value in fn['parameters'].items()]
        allowed_funcs.add(fn['name'])
        func_ids = self.model.encode(fn['name'])[0].tolist()
        node = self.func_trie
        for tid in func_ids:
            if tid not in node:
                node[tid] = {}
            node = node[tid]
        node["__END__"] = fn['name']
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
      tensor_ids = self.model.encode(prompt)
      input_ids = tensor_ids[0].tolist()
      text = json_start

      done_json = 0
      open_braces = 1
      closed_braces = 0
      func_name = ""
      parameters = ""

      i = 0
      next_arg = 1
      trie_node = self.func_trie
      while True:
        logits = self.model.get_logits_from_input_ids(input_ids)
        next_token_id = int(np.argmax(logits))

        print(f"Next Token id : {next_token_id}")

        if self.extract_func_name:
          allowed_ids = [tid for tid in trie_node if tid != "__END__"]

          if next_token_id not in self.func_vocab:
            print("Wrong!\n")
            mask = np.full(len(logits), -np.inf)

            for tid in allowed_ids:
                        mask[tid] = 0.0
            mask[self.func_vocab]
            constrained_logits = logits + mask
            next_token_id = int(np.argmax(constrained_logits))
            print(f"Next Token id : {next_token_id}")

          current_text = self.model.decode([next_token_id])
          trie_node = trie_node[next_token_id]
          func_name += current_text

          if "__END__" in trie_node:
            self.func_name = trie_node["__END__"]
            self.extract_func_name = False  # done extracting name
            trie_node = self.func_trie 

        else:
          current_text = self.model.decode([next_token_id])
          
        print(f"\nToken: {current_text}")

        return
        if current_text.endswith(','):
          current_text += forced_after_comma
          input_ids.extend(after_comma_id)
        
        if not done_json:
          open_braces += current_text.count("{")
          closed_braces += current_text.count("}")

          injected = False

          if not self.extract_func_name:
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
              par_tensor = self.model.encode(par_str)
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

























