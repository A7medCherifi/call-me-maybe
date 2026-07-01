import numpy as np
import re
import ast
import time
from llm_sdk import Small_LLM_Model


class Model():
    def __init__(self, manager):
        self.model = Small_LLM_Model()
        self.functions_data = ""
        self.output_text = ""
        self.input_str = ""
        self.func_name = ""
        self.parameters = ""
        self.current_token = ""
        self.manager = manager
        self.resources = dict()
        self.all_prompts = list()

        self.input_ids = None
        self.extract_func_name = True
        self.inject_par = False
        self.injected = False

        self.prompt = ""

    def _stage_of_prompt(self):
        self.prompt = f"""
        Functions Data: \
            {self.functions_data} \
        Extract the function name from Functions Data and parameters as a valid JSON object. \
        Example: \
            Input text: What is the sum of 2 and 3? \
            JSON output: \"name\": \"fn_add_numbers\", \"parameters\": {{"a": 2.0, "b": 3.0}}.\
        Rules: \
            1. If a parameter type is a Number cast it to a float. \
            2. Output ONLY the raw JSON. \
        Input Text: {self.input_str} \
        JSON output: \
        """
        json_start = "{" + f'"prompt": "{self.input_str}", "name": "'
        self.prompt += json_start
        self.output_text = json_start

        print(self.functions_data)
        tensor_ids = self.model.encode(self.prompt)
        self.input_ids = tensor_ids[0].tolist()


    def _stage_of_name(self):
        if self.extract_func_name:
            if '"' not in self.current_token:
                self.func_name += self.current_token
            else:
                self.func_name += self.current_token.split('"')[0]
                self.output_text += self.current_token
                self.extract_func_name = False


    def _stage_of_parameter(self, i):
        if not self.inject_par:
            inject_parameter_str = '"parameters": {'
            parameter_tensor = self.model.encode(inject_parameter_str)
            parameter_ids = parameter_tensor[0].tolist()
            self.input_ids.extend(parameter_ids)
            self.output_text += inject_parameter_str
            self.injected = True
            self.inject_par = True

        if self.inject_par:
            if self.resources[self.func_name][i][1] == 'string':
                par_str = f'"{self.resources[self.func_name][i][0]}": "'
            else:
                par_str = f'"{self.resources[self.func_name][i][0]}": '
            par_tensor = self.model.encode(par_str)
            par_ids = par_tensor[0].tolist()
            self.input_ids.extend(par_ids)
            self.output_text += par_str
            self.injected = True
            self.parameters += par_str


    def _extract_data_from_input(self):
        for fn in self.manager.definition_functions:
            self.functions_data += f"Name: {fn['name']} | Parameters: {fn['parameters']}\n"
            self.resources[fn['name']] = [(name, value['type']) for name, value in fn['parameters'].items()]
        for element in self.manager.prompts_calling:
            self.all_prompts.append(element['prompt'])


    def run_model(self, input_str):
        self._extract_data_from_input()
        start = time.perf_counter()
        after_comma_id = self.model.encode(' ')[0].tolist()

        # for input in all_prompts:
        # ===== Stage of Prompt ===== 
        self.input_str = input_str
        self._stage_of_prompt()

        i = 0
        next_arg = 1
        done_json = 0
        open_braces = 1
        closed_braces = 0

        while True:
            logits = self.model.get_logits_from_input_ids(self.input_ids)
            next_token_id = int(np.argmax(logits))

            self.current_token = self.model.decode([next_token_id])

            # check if the current has ',' forced injection ' ' to it
            if self.current_token.endswith(','):
                self.current_token += ' '
                self.input_ids.extend(after_comma_id)

            # keep counting braces if still json didnt finish
            open_braces += self.current_token.count("{")
            closed_braces += self.current_token.count("}")

            self.injected = False

            # ===== Stage of Name ===== 
            self._stage_of_name()
            
            # ===== Stage of Parameters ===== 
            if not self.extract_func_name and i < len(self.resources[self.func_name]):
                if next_arg:
                    if not self.inject_par:
                        open_braces += 1
                    self._stage_of_parameter(i)
                    i += 1
                    next_arg = 0
                else:
                    self.parameters += self.current_token

                if self.parameters.count(',') == i:
                    next_arg = 1

            # add the current token to the output text
            if not self.injected:
                self.input_ids.append(next_token_id)
                self.output_text += self.current_token

            # check if the json done
            if open_braces == closed_braces:
                if self.inject_par:
                    self.output_text, braces, _ = self.output_text.rpartition('}}')
                    self.output_text += braces
                    done_json = 1

            if done_json:
                break

            print(self.output_text)

        print("\n#################################################\n")

        elapsed = time.perf_counter() - start
        print(f"Time: {elapsed / 60:.2f}")








