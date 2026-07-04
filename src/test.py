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
        self.func_name = ""
        self.parameters = ""
        self.current_token = ""
        self.resources = dict()
        self.all_prompts = list()
        self.vocab = dict()

        self.input_ids = None
        self.const_prompt_ids = None
        self.extract_func_name = True
        self.inject_par = False
        self.injected = False

    def __get_valid_parameters(self, par_type):
        valid_vocab = dict()
        if par_type == 'string':
            pass
        else:
            vocab = ['.', '-', '+']
            for i in range(10):
                vocab.append(str(i))
            for element in vocab:
                element_id = self.model.encode(element)[0].tolist()
                valid_vocab[element] = element_id
        return valid_vocab

    def _encode_constant_prompt(self):
        const_prompt = f"""
            Functions Data: \
                {self.functions_data} \
            Extract the function name from Functions Data and parameters as a valid JSON object. \
            Example: \
                Input text: What is the sum of 2 and 3? \
                JSON output: \"name\": \"fn_add_numbers\", \"parameters\": {{"a": 2.0, "b": 3.0}}.\
            Rules: \
                1. If a parameter type is a Number cast it to a float. \
                2. If a parameter type is an Integer keep it as a valid integer not float.
                2. Output ONLY the raw JSON. \
            Input Text: 
        """
        self.const_prompt_ids = self.model.encode(const_prompt)[0].tolist()

    def _stage_of_prompt(self):
        self.prompt = f"""{self.input_str} \
        JSON output: \
        """
        json_start = "{" + f'"prompt": "{self.input_str}", "name": "'
        self.prompt += json_start
        self.output_text = json_start

        tensor_ids = self.model.encode(self.prompt)
        self.input_ids = copy.deepcopy(self.const_prompt_ids)
        self.input_ids += tensor_ids[0].tolist()


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
            func_ids = self.model.encode(fn['name'])[0].tolist()
            self.vocab[fn['name']] = func_ids

        for element in self.manager.prompts_calling:
            self.all_prompts.append(element['prompt'])

    # def _mask_unwanted_tokens(self, logits, found_name, valid_vocab, stage):
    #     if stage == 1:
    #         return self._handle_func_name(logits, found_name, valid_vocab)
            
    #     elif stage == 2:
    #         return self._handle_parameters(logits, found_name, valid_vocab)

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
                        print(f"\nFound it: {next_token_id}, Vocab: {valid_vocab}\n")
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

    def _handle_parameters(self, logits, par_type, valid_vocab):
        """
        Add constrained decoding for the parameters, by implementing those:
            1. Check the type of parameter if its string it must start with '"' and ends with it.
            2. Check the type of parameter if its integer it must handle integers only and mask other tokens.
            3. Check the type of parameter if its Number it must be a float ends with .\+ (0-9).
            4. Check for extra quots or spaces.
            5. if '{' or '}' in the value of parameter you should now count it as a real braces of the json, it must be a char only.
        """
        if par_type in ['number', 'integer']:
            valid_vocab = self.__get_valid_parameters(par_type)
            mask = np.full_like(logits, )
            
        elif par_type == 'string':
            pass
        

    def run_model(self, input_str):
        self._extract_data_from_input()
        start = time.time()
        self._encode_constant_prompt()
        after_comma_id = self.model.encode(' ')[0].tolist()

        # ===== Stage of Prompt ===== 
        # for input in self.manager.prompts_calling:
        self.input_str = input_str
        self._stage_of_prompt()

        i = 0
        next_arg = 1
        done_json = 0
        open_braces = 1
        closed_braces = 0

        self.func_name = ""
        self.parameters = ""

        self.extract_func_name = True
        self.inject_par = False
        self.injected = False

        found_name = False
        valid_vocab = copy.deepcopy(self.vocab)
        stage = 1

        while True:
            logits = self.model.get_logits_from_input_ids(self.input_ids)
            print(logits)

            if stage == 1:
                next_token_id, found_name, valid_vocab = self._handle_func_name(
                    logits,
                    found_name,
                    valid_vocab
                )
            else:
                next_token_id, found_name, valid_vocab = self._handle_parameters(
                    logits,
                    found_name,
                    valid_vocab
                )


            if found_name:
                self.current_token = self.model.decode(next_token_id)
            else:
                self.current_token = self.model.decode([next_token_id])

            if self.current_token.endswith(','):
                self.current_token += ' '
                self.input_ids.extend(after_comma_id)

            if not done_json:
                open_braces += self.current_token.count("{")
                closed_braces += self.current_token.count("}")

                self.injected = False

                # ===== Stage of Name ===== 
                self._stage_of_name()
                
                # ===== Stage of Parameters ===== 
                if not self.extract_func_name and i < len(self.resources[self.func_name]):
                    stage = 2
                    if next_arg:
                        if not self.inject_par:
                            open_braces += 1
                        self._stage_of_parameter(i)
                        i += 1
                        next_arg = 0

                    if self.parameters.count(',') == i:
                        next_arg = 1

                if not self.injected:
                    if found_name:
                        self.input_ids.extend(next_token_id)
                        found_name = False
                    else:
                        self.input_ids.append(next_token_id)
                    self.output_text += self.current_token
                    if self.inject_par:
                        if '}' in self.current_token:
                            self.current_token, _, _ = self.current_token.partition('}')
                        self.parameters += self.current_token

                if open_braces == closed_braces:
                    if self.inject_par:
                        self.output_text, braces, _ = self.output_text.rpartition('}}')
                        self.output_text += braces
                        done_json = 1
            else:
                break

            print(self.output_text)
            print(f"Parameters: {self.parameters}\n")

        print("\n#################################################\n")
            
        end = time.time()
        print(f"Time: {(end - start) / 60:.2f}")








