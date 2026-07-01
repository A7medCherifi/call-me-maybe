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
        self.input = ""
        self.func_name = ""
        self.current_token = ""
        self.manager = manager
        self.resources = dict()
        self.all_prompts = list()

        self.prompt = f"""
        Extract the function name and parameters as a valid JSON object. \
        Functions: \
        {self.functions_data} \
        Example: \
        Input text: What is the sum of 2 and 3? \
        JSON output: \"name\": \"fn_add_numbers\", \"parameters\": {{"a": 2.0, "b": 3.0}}.\
        Rules: \
        1. If a parameter type is a Number cast it to a float. \
        2. Output ONLY the raw JSON. \
        Input Text: {self.input} \
        JSON output: \
        """

    def _stage_of_prompt(self):
        json_start = "{" + f'"prompt": "{self.input}", "name": "'
        self.prompt += json_start
        self.output_text = json_start



    def _stage_of_name(self):
        pass



    def _stage_of_parameter(self):
        pass



    def _extract_data_from_input(self):
        for fn in self.manager.definition_functions:
            self.functions_data += f"Name: {fn['name']} | Parameters: {fn['parameters']}\n"
            self.resources[fn['name']] = [(name, value['type']) for name, value in fn['parameters'].items()]

        for element in self.manager.prompts_calling:
            self.all_prompts.append(element['prompt'])





    def run_model(self, manager, input):
        start = time.perf_counter()

        inject_parameter_str = '"parameters": {'
        parameter_tensor = self.model.encode(inject_parameter_str)
        parameter_ids = parameter_tensor[0].tolist()

        forced_after_comma = ' '
        after_comma_tensor = self.model.encode(forced_after_comma)
        after_comma_id = after_comma_tensor[0].tolist()


        # for input in all_prompts:
        self.input = input
        
        tensor_ids = self.model.encode(self.prompt)
        input_ids = tensor_ids[0].tolist()

        done_json = 0
        open_braces = 1
        closed_braces = 0
        parameters = ""

        i = 0
        next_arg = 1
        extract_func_name= True
        while True:
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token_id = int(np.argmax(logits))

            self.current_token = self.model.decode([next_token_id])

            # check if the current has ',' forced injection ' ' to it
            if self.current_token.endswith(','):
                self.current_token += forced_after_comma
                input_ids.extend(after_comma_id)

            # keep counting braces if still json didnt finish
            if not done_json:
                self._stage_of_name()
                open_braces += self.current_token.count("{")
                closed_braces += self.current_token.count("}")
                injected = False

                # extract function name and inject parameter str
                if extract_func_name:
                    if '"' not in self.current_token:
                        self.func_name += self.current_token
                    else:
                        self.func_name += self.current_token.split('"')[0]
                        extract_func_name = False
                        output_text += self.current_token

                        input_ids.extend(parameter_ids)
                        output_text += inject_parameter_str
                        open_braces += 1
                        injected = True

                # inject the parameter keys
                if 'parameters' in output_text and i < len(self.resources[self.func_name]):
                    if next_arg:
                        if self.resources[self.func_name][i][1] == 'string':
                            par_str = f'"{self.resources[self.func_name][i][0]}": "'
                        else:
                            par_str = f'"{self.resources[self.func_name][i][0]}": '
                            par_tensor = self.model.encode(par_str)
                            par_ids = par_tensor[0].tolist()

                            input_ids.extend(par_ids)
                            output_text += par_str
                            parameters += par_str
                            i += 1
                            next_arg = 0
                            injected = True

                    else:
                        parameters += self.current_token

                    if parameters.count(',') == i:
                        next_arg = 1

                # add the current token to the output text
                if not injected:
                    input_ids.append(next_token_id)
                    output_text += self.current_token

                # check if the json done
                if open_braces == closed_braces:
                    if 'parameters' in output_text:
                        output_text, braces, _ = output_text.rpartition('}}')
                        output_text += braces
                        done_json = 1

            else:
                break
            print(output_text)

        print("\n#################################################\n")

        elapsed = time.perf_counter() - start
        print(f"Time: {elapsed / 60:.2f}")








