import numpy as np
import copy
import time
import json
from typing import Any, Dict, List, Tuple, Union

from llm_sdk.llm_sdk import Small_LLM_Model


class Model:
    """A class representing the function-calling LLM execution engine.\

    This class handles the token-by-token constrained decoding loop, injecting\
    structured grammar constraints to guide the llm model into producing\
    guaranteed valid answer and valid JSON output matching predefined schemas.\
    """
    def __init__(self, manager: Any) -> None:
        """Initializes the Model with necessary states.\

        Args:\
            manager (Any): The manager that handles input files.\
        """
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.manager: Any = manager

        self.functions_data: str = ""
        self.output_text: str = ""
        self.print_text: str = ""
        self.input_str: str = ""
        self.prompt: str = ""
        self.func_name: str = ""
        self.current_token: str = ""
        self.par_value: str = ""

        self.number_value: str = ""

        self.vocab: Dict[str, List[int]] = dict()
        self.resources: Dict[str, List[Tuple[str, str]]] = dict()
        self.all_prompts: List[str] = list()
        self.invalid_tokens: List[Any] = list()
        self.output: List[Dict[str, Any]] = list()

        self.input_ids: List[int] = list()
        self.const_prompt_ids: List[int] = list()

    def __get_valid_digits(self, par_type: str) -> List[int]:
        """
        Get valid vocabulary for Digits

        Args:
            par_tupe (str): parameter type

        Returns:
            list: of encoded valid vocab
        """
        valid_vocab: List[int] = list()
        vocab: List[str] = ['+', '-', ',', '}}'] + [str(i) for i in range(10)]
        if par_type == 'number':
            vocab.append('.')
        for element in vocab:
            element_id: List[int] = self.model.encode(element)[0].tolist()
            valid_vocab.extend(element_id)
        return valid_vocab

    def __get_valid_boolean(self) -> List[int]:
        """Get valid vocabulary for boolean

        Returns:
            list: of encoded valid vocab
        """
        valid_vocab: List[int] = list()
        vocab: List[str] = ['true', 'false']
        for element in vocab:
            element_id: List[int] = self.model.encode(element)[0].tolist()
            valid_vocab.extend(element_id)
        return valid_vocab

    def _encode_constant_prompt(self) -> None:
        """
        Encode the fixed prompt that has all the instructions
        of what to do that llm needs
        """
        const_prompt: str = f"""
            Functions Data:\
                {self.functions_data}\
            Extract the function name from Functions Data and valid parameters\
                as a valid JSON object.\
            Examples: \
                Input text: What is the sum of 2 and 3? \
                JSON output: \"name\": \"fn_add_numbers\",
                \"parameters\": {{"a": 2.0, "b": 3.0}}.\
            Rules: \
                1. If a parameter type is a number cast it to a float always.\
                3. Output ONLY the raw JSON. \
                4. If a parameter key is regex the value\
                must be a valid REGEX sequence of characters\
            Input Text: \
        """
        self.const_prompt_ids = self.model.encode(const_prompt)[0].tolist()

    def _stage_of_prompt(self) -> None:
        """
        Encode the prompt of the user and inject \
        the prompt into the output string
        """
        self.prompt = f"""{self.input_str} \
        JSON output: \
        """
        input_str: str = json.dumps(self.input_str)
        json_start: str = "{" + f'"prompt": {input_str}, "name": "'
        self.prompt += json_start
        self.output_text = json_start
        self.print_text = json_start

        self.input_ids = copy.deepcopy(self.const_prompt_ids)
        self.input_ids.extend(self.model.encode(self.prompt)[0].tolist())

    def _stage_of_name(self, found_name: bool, next_token_id:
                       Union[int, List[int]], stage: int) -> int:
        """Extract the function name from the token, and inject the spliter
        token to the end of func name.

        Args:
            found_name (boolean): did you found the full name or not.
            next_token_id (int): token id
            stage (int): represent the current stage.

        Returns:
            int: stage
        """
        if found_name:
            assert isinstance(next_token_id, list)
            self.current_token = self.model.decode(next_token_id)
            splited_token: str = self.current_token.split('"')[0].strip()
            self.func_name += splited_token
            spliter_id: List[int] = self.model.encode('", ')[0].tolist()
            self.input_ids.extend(next_token_id)
            self.input_ids.extend(spliter_id)
            self.output_text += splited_token + '", '
            self.print_text += splited_token + '", '
            stage = 2
        else:
            assert isinstance(next_token_id, int)
            self.current_token = self.model.decode([next_token_id])
            self.func_name += self.current_token
            self.input_ids.append(next_token_id)
            self.output_text += self.current_token
            self.print_text += self.current_token
        return stage

    def _stage_of_inject_parameter(self, stage: int) -> int:
        """Inject the parameter string

        Args:
            stage (int): represent the current stage.

        Returns:
            int: stage
        """
        inject_parameter_str: str = '"parameters": {'
        parameter_ids: List[int] = self.model.encode(
            inject_parameter_str)[0].tolist()
        self.input_ids.extend(parameter_ids)
        self.output_text += inject_parameter_str
        self.print_text += inject_parameter_str
        stage = 3
        return stage

    def _stage_of_inject_key(self, i: int, stage: int) -> int:
        """Inject the parameter key or closed braces if its the end of json

        Args:
            i (int): represent the current parameter
            stage (int): represent the current stage.

        Returns:
            int: stage
        """
        if i < len(self.resources[self.func_name]):
            par_type: str = self.resources[self.func_name][i][1]
            if par_type == 'string':
                par_str: str = f'"{self.resources[self.func_name][i][0]}": "'
            else:
                par_str = f'"{self.resources[self.func_name][i][0]}": '
            par_ids: List[int] = self.model.encode(par_str)[0].tolist()
            self.input_ids.extend(par_ids)
            self.output_text += par_str
            self.print_text += par_str
            stage = 4
            return stage
        else:
            self.output_text += '}}'
            self.print_text += '}}'
            stage = 5
            return stage

    def _extract_data_from_input(self) -> None:
        """
        Extract the data that we need from the manager
        after parsing.
        """
        for fn in self.manager.definition_functions:
            self.functions_data += f"\
Name: {fn['name']} | Parameters: {fn['parameters']}\n"
            self.resources[fn['name']] = []
            for name, value in fn['parameters'].items():
                self.resources[fn['name']].append((name, value['type']))
            func_ids: List[int] = self.model.encode(fn['name'])[0].tolist()
            self.vocab[fn['name']] = func_ids

        for element in self.manager.prompts_calling:
            self.all_prompts.append(element['prompt'])

    def _handle_func_name(
        self, logits: Any, found_name: bool, valid_vocab: Dict[str, List[int]]
    ) -> Tuple[Union[int, List[int]], bool, Dict[str, List[int]]]:
        """Constrained decoding for function name, to make sure
        that next token is on function names.

        Args:
            logits (list): raw of scores of the whole vocab of llm
            found_name (boolean): did you found the full name or not.
            valid_vocab (dict): copy of valid function names with their ids

        Returns:
            tuple: (next_token_id, found_name, valid_vocab)
        """
        funcs_to_remove: List[str] = []
        matched: bool = False
        while True:
            next_token_id: int = int(np.argmax(logits))
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
                        next_token_res: Union[int, List[int]] = valid_vocab[
                            func]
                        del valid_vocab[func]
                        break
                    return (next_token_res, found_name, valid_vocab)
                for func in funcs_to_remove:
                    del valid_vocab[func]
                for func in valid_vocab:
                    valid_vocab[func].pop(0)
                break
            else:
                logits[next_token_id] = -float('inf')
                funcs_to_remove = []
        return (next_token_id, found_name, valid_vocab)

    def _handle_parameters(self, logits: Any, par_type: str) -> List[int]:
        """Constrained decoding of the parameters.

        Args:
            logits (list): raw of scores of the whole vocab of llm.
            par_type (str): Parameter type.

        Returns:
            list(): [next_token_id]
        """
        if par_type in ['number', 'integer']:
            valid_vocab: List[int] = self.__get_valid_digits(par_type)
            logits = np.array(logits)
            mask: np.ndarray = np.full_like(logits, -float('inf'))
            mask[valid_vocab] = logits[valid_vocab]
            return [int(np.argmax(mask))]
        elif par_type == "boolean":
            valid_vocab = self.__get_valid_boolean()
            logits = np.array(logits)
            mask = np.full_like(logits, -float('inf'))
            mask[valid_vocab] = logits[valid_vocab]
            return [int(np.argmax(mask))]
        else:
            logits = np.array(logits)
            mask = copy.deepcopy(logits)
            if not self.invalid_tokens:
                tab_id: List[int] = self.model.encode('\t')[0].tolist()
                self.invalid_tokens.append(tab_id)
                newline_id: List[int] = self.model.encode('\n')[0].tolist()
                self.invalid_tokens.append(newline_id)
            mask[self.invalid_tokens] = -float('inf')
            return [int(np.argmax(mask))]

    def _check_if_value_done(self, par_type: str,
                             input_str: str) -> Tuple[bool, str]:
        """Track the parameter value if its done.

        Args:
            par_type (str): Parameter type.
            input_str (str): the input prompt string.

        Returns:
            tuple(): (par_finish, token_to_add)
        """
        token_to_add: str = self.current_token
        par_finish: bool = False
        if par_type == 'boolean':
            if token_to_add in ['true', 'false']:
                par_finish = True
        if par_type == 'string':
            if '"' in token_to_add and '\\"' not in token_to_add:
                token_to_add = token_to_add.split('"')[0]
                par_finish = True
        else:
            if ',' in self.current_token or '}}' in self.current_token:
                par_finish = True
                token_to_add = ""

        if not par_finish and len(self.par_value) >= len(input_str):
            par_finish = True
        return (par_finish, token_to_add)

    def _inject_par_spliter(self, token_to_add: str, par_type: str,
                            i: int) -> Tuple[int, int]:
        """inject parameter spliters or the end braces if the parameters done
        and move the stage.

        Args:
            token_to_add (str): copy string of the token.
            par_type (str): Parameter type.
            i (int): Current indexed parameter.

        Returns:
            tuple(): (stage, i)
        """
        if token_to_add:
            if par_type != 'string':
                self.output_text += token_to_add
            self.print_text += token_to_add
            self.par_value += token_to_add
            token_to_add_id: List[int] = self.model.encode(
                token_to_add)[0].tolist()
            self.input_ids.extend(token_to_add_id)

        i += 1
        if i < len(self.resources[self.func_name]):
            sep_str: str = ""
            if par_type == 'string':
                sep_str = '", '
            else:
                sep_str = ', '

            sep_id: List[int] = self.model.encode(sep_str)[0].tolist()
            self.input_ids.extend(sep_id)
            if par_type == 'number':
                value_f: float = float(self.number_value)
                self.output_text += str(value_f)
                self.number_value = ""

            elif par_type == 'integer':
                value_i: int = int(self.number_value)
                self.output_text += str(value_i)
                self.number_value = ""

            elif par_type == 'string':
                res = json.dumps(self.par_value)
                self.output_text += res[1:-1]
            self.output_text += sep_str
            self.print_text += sep_str
            self.par_value = ""
            stage: int = 3

        else:
            if par_type == 'number':
                value_f = float(self.number_value)
                self.output_text += str(value_f)
                self.number_value = ""
            elif par_type == 'integer':
                value_i = int(self.number_value)
                self.output_text += str(value_i)
                self.number_value = ""
            elif par_type == 'string':
                res = json.dumps(self.par_value)
                self.output_text += res[1:-1]

            end_str: str = ""
            if par_type == 'string':
                end_str = '"}}'
            else:
                end_str = '}}'
            sep_id = self.model.encode(end_str)[0].tolist()
            self.input_ids.extend(sep_id)
            self.output_text += end_str
            self.print_text += end_str
            self.par_value = ""
            stage = 5
        return (stage, i)

    def run_model(self) -> List[Dict[str, Any]]:
        """
        Executes the pipeline loop processing input\
        and generate the valid json.\

        Returns:\
            list(): A collection of generated json.\
        """
        self._extract_data_from_input()
        start: float = time.time()
        self._encode_constant_prompt()

        for input_str in self.all_prompts:
            self.input_str = input_str
            self._stage_of_prompt()
            self.func_name = ""
            valid_vocab: Dict[str, List[int]] = copy.deepcopy(self.vocab)

            i: int = 0
            stage: int = 1
            found_name: bool = False

            while stage != 5:
                logits: List[float] = self.model.get_logits_from_input_ids(
                    self.input_ids)

                if stage == 1:
                    result: Tuple[int | list[int], bool, Dict[str, List[int]]
                                  ] = self._handle_func_name(
                        logits, found_name, valid_vocab
                    )
                    next_token_id_res, found_name, valid_vocab = result
                    stage = self._stage_of_name(found_name, next_token_id_res,
                                                stage)
                elif stage == 2:
                    stage = self._stage_of_inject_parameter(stage)

                elif stage == 3:
                    stage = self._stage_of_inject_key(i, stage)

                elif stage == 4:
                    par_type: str = self.resources[self.func_name][i][1]
                    next_token_ids: List[int] = self._handle_parameters(
                        logits, par_type)
                    self.current_token = self.model.decode(next_token_ids)

                    par_finish, token_to_add = self._check_if_value_done(
                        par_type, input_str)
                    if par_finish:
                        stage, i = self._inject_par_spliter(
                            token_to_add, par_type, i)
                    else:
                        if isinstance(next_token_ids, list):
                            self.input_ids.extend(next_token_ids)
                        else:
                            self.input_ids.append(next_token_ids)

                        if not self.par_value and (par_type == 'string' or
                                                   par_type == 'boolean'):
                            if par_type == 'boolean':
                                self.output_text += self.current_token.strip()
                            self.print_text += self.current_token.strip()
                            self.par_value += self.current_token.strip()
                        elif par_type == 'string' or par_type == 'boolean':
                            if par_type == 'boolean':
                                self.output_text += self.current_token
                            self.print_text += self.current_token
                            self.par_value += self.current_token
                        else:
                            self.par_value += self.current_token
                            self.print_text += self.current_token
                            self.number_value += self.current_token

                print(f"{self.print_text}")

            print(f"\n{self.output_text}")
            data: Dict[str, Any] = json.loads(self.output_text)
            self.output.append(data)
            print("\n#################################################\n")
        end: float = time.time()
        print(f"Time: {(end - start):.2f}")
        return self.output
