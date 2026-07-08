import numpy as np
import copy
import re
import ast
import time
from llm_sdk import Small_LLM_Model


class Model():
    def __init__(self, manager):
        self.model = Small_LLM_Model()
        self.manager = manager
        
        self.functions_data = ""
        self.output_text = ""
        self.input_str = ""
        self.prompt = ""
        self.par_type = ""
        self.func_name = ""
        self.parameters = ""
        self.current_token = ""

        self.par_value = ""

        self.vocab = dict()
        self.resources = dict()
        self.all_prompts = list()
        self.valid_digits = list()

        self.input_ids = None
        self.const_prompt_ids = None
        self.extract_func_name = True
        self.inject_par = False
        self.injected = False

        self.isvalue = False
        self.par_count = 0

    def __get_valid_digits(self):
        valid_vocab = list()
        vocab = ['+', '-', ',', '}}', '.'] + [str(i) for i in range(10)]
        for element in vocab:
            element_id = self.model.encode(element)[0].tolist()
            valid_vocab.extend(element_id)
        return valid_vocab

    def _encode_constant_prompt(self):
        const_prompt = f"""
            Functions Data: \
                {self.functions_data} \
            Extract the function name from Functions Data and valid parameters as a valid JSON object. \
            Example: \
                Input text: What is the sum of 2 and 3? \
                JSON output: \"name\": \"fn_add_numbers\", \"parameters\": {{"a": 2.0, "b": 3.0}}.\
            Rules: \
                1. If a parameter type is a 'number' cast it to a float. \
                2. If a parameter type is an 'integer' keep it as a valid integer NOT float. \
                3. If a parameter key is regex extract a valid regex value always. \
                4. Output ONLY the raw JSON. \
            Input Text: \
        """
        self.const_prompt_ids = self.model.encode(const_prompt)[0].tolist()

    def _stage_of_prompt(self):
        self.prompt = f"""{self.input_str} \
        JSON output: \
        """
        json_start = "{" + f'"prompt": "{self.input_str}", "name": "'
        self.prompt += json_start
        self.output_text = json_start

        self.input_ids = copy.deepcopy(self.const_prompt_ids)
        self.input_ids += self.model.encode(self.prompt)[0].tolist()

    def _stage_of_name(self, found_name, next_token_id, stage):
        if found_name:
            self.current_token = self.model.decode(next_token_id)
            splited_token = self.current_token.split('"')[0].strip()
            self.func_name += splited_token
            spliter_id = self.model.encode('", ')[0].tolist()
            self.input_ids.extend(next_token_id)
            self.input_ids.extend(spliter_id)
            self.output_text += splited_token + '", '
            stage = 2
        else:
            self.current_token = self.model.decode([next_token_id])
            self.func_name += self.current_token
            self.input_ids.append(next_token_id)
            self.output_text += self.current_token
        return stage

    def _stage_of_inject_parameter(self, stage):
        inject_parameter_str = '"parameters": {'
        parameter_ids = self.model.encode(inject_parameter_str)[0].tolist()
        self.input_ids.extend(parameter_ids)
        self.output_text += inject_parameter_str
        stage = 3
        return stage

    def _stage_of_inject_key(self, i, stage):
        if i < len(self.resources[self.func_name]):
            par_type = self.resources[self.func_name][i][1]
            if par_type == 'string':
                par_str = f'"{self.resources[self.func_name][i][0]}": "'
            else:
                par_str = f'"{self.resources[self.func_name][i][0]}": '
            par_ids = self.model.encode(par_str)[0].tolist()
            self.input_ids.extend(par_ids)
            self.output_text += par_str
            stage = 4
            return stage
        else:
            self.output_text += '}}'
            stage = 5
            return stage

    def _extract_data_from_input(self):
        for fn in self.manager.definition_functions:
            self.functions_data += f"Name: {fn['name']} | Parameters: {fn['parameters']} | Description of function: {fn['description']}\n"
            self.resources[fn['name']] = [(name, value['type']) for name, value in fn['parameters'].items()]
            func_ids = self.model.encode(fn['name'])[0].tolist()
            self.vocab[fn['name']] = func_ids

        for element in self.manager.prompts_calling:
            self.all_prompts.append(element['prompt'])

    def _handle_func_name(self, logits, found_name, valid_vocab):
        funcs_to_remove = []
        matched = False
        while True:
            next_token_id = int(np.argmax(logits))
            if not valid_vocab:
                break
            for name, ids in valid_vocab.items():
                if len(ids) > 0 and ids[0] == next_token_id:
                    matched = True
                else:
                    funcs_to_remove.append(name)
            if matched:
                if len(valid_vocab) == 1:
                    found_name = True
                    for func in valid_vocab:
                        next_token_id = valid_vocab[func]
                        del valid_vocab[func]
                        break
                    break
                for func in funcs_to_remove:
                    del valid_vocab[func]
                for func in valid_vocab:
                    valid_vocab[func].pop(0)
                break
            else:
                logits[next_token_id] = -float('inf')
                funcs_to_remove = []
        return (next_token_id, found_name, valid_vocab)

    def _handle_parameters(self, logits, par_type):        
        if par_type in ['number', 'integer']:
            valid_vocab = self.valid_digits
            logits = np.array(logits)
            mask = np.full_like(logits, -float('inf'))
            mask[valid_vocab] = logits[valid_vocab]
            return int(np.argmax(mask))
        else:
            return int(np.argmax(logits))
        

    def run_model(self, input_str):
        self._extract_data_from_input()
        start = time.time()
        self.valid_digits = self.__get_valid_digits()
        self._encode_constant_prompt()
        # after_comma_id = self.model.encode(' ')[0].tolist()

        # ===== Stage of Prompt ===== 
        for inputs in self.manager.prompts_calling:
            input_str = next(iter(inputs.values()))
            self.input_str = input_str
            self._stage_of_prompt()

            i = 0
            next_arg = 1
            done_json = 0
            open_braces = 1
            closed_braces = 0

            self.func_name = ""
            self.parameters = ""
            self.par_type = ""
            value_str = ""

            self.extract_func_name = True
            self.inject_par = False
            self.injected = False

            found_name = False
            valid_vocab = copy.deepcopy(self.vocab)
            stage = 1

            while stage != 5:
                logits = self.model.get_logits_from_input_ids(self.input_ids)

                if stage == 1:
                    next_token_id, found_name, valid_vocab = self._handle_func_name(logits, found_name, valid_vocab)
                    stage = self._stage_of_name(found_name, next_token_id, stage)

                elif stage == 2:
                    stage = self._stage_of_inject_parameter(stage)

                elif stage == 3:
                    stage = self._stage_of_inject_key(i, stage)

                elif stage == 4:
                    par_type = self.resources[self.func_name][i][1]
                    next_token_id = self._handle_parameters(logits, par_type)
                    self.current_token = self.model.decode([next_token_id])

                    par_finish = False
                    token_to_add = self.current_token
                    if par_type == 'string':
                        if '"' in token_to_add:
                            token_to_add = token_to_add.split('"')[0]
                            par_finish = True
                    else:
                        if ',' in self.current_token or '}' in self.current_token:
                            par_finish = True
                            token_to_add = ""

                    if not par_finish and len(self.par_value) >= len(input_str):
                        par_finish = True

                    if par_finish:
                        if token_to_add:
                            self.output_text += token_to_add
                            token_to_add_id = self.model.encode(token_to_add)[0].tolist()
                            self.input_ids.extend(token_to_add_id)
                        i += 1
                        if i < len(self.resources[self.func_name]):
                            sep_str = ""
                            if par_type == 'string':
                                sep_str = '", '
                            else:
                                sep_str = ', '
                            sep_id = self.model.encode(sep_str)[0].tolist()
                            self.input_ids.extend(sep_id)
                            self.output_text += sep_str
                            stage = 3
                        else:
                            end_str = ""
                            if par_type == 'string':
                                end_str = '"}}'
                            else:
                                end_str = '}}'
                            sep_id = self.model.encode(end_str)[0].tolist()
                            self.input_ids.extend(sep_id)
                            self.output_text += end_str
                            stage = 5
                    else:
                        self.input_ids.append(next_token_id)
                        self.output_text += self.current_token
                        self.par_value += self.current_token

                print(self.output_text)

            print("\n#################################################\n")
            
        end = time.time()
        print(f"Time: {(end - start) / 60:.2f}")




"""

    Fixed the token injection that you do, it miss it out if you give ',' in the prompt, 

    Fixed double '""' and find a way to make sure that he will stop and return a valid json

    3awed 9ad constrained decoding l digits cause ma3jbnich hadik ghir mslka makhdamach mzn



    moraha handli l input dyal user ou kifach ghadi it3amel m3ah make some code rules

    bach maygeneration chi tkhwira li tkhower lik hadchi

    thats it.

    Add constrained decoding for the parameters, by implementing those:
            1. Check the type of parameter if its string it must start with '"' and ends with it.
            2. Check the type of parameter if its integer it must handle integers only and mask other tokens.
            3. Check the type of parameter if its Number it must be a float ends with .\+ (0-9).
            4. Check for extra quots or spaces.
            5. if '{' or '}' in the value of parameter you should now count it as a real braces of the json, it must be a char only.


"""



