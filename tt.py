import numpy
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()


def test_model(manager, input):

  functions = ""
  for fn in manager.definition_functions:
          functions += f"Name: {fn['name']} | Parameters: {fn['parameters']}\n"

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
  test_tensor = model.encode(prompt)
  test_ids = test_tensor[0].tolist()

  text = json_start
  for _ in range(50):
    logits = model.get_logits_from_input_ids(test_ids)
    next_token = int(numpy.argmax(logits))
    test_ids.append(next_token)

    token_str = model.decode([next_token])
    text += token_str
    if 'parameters' in text and '}}' in text:
      text, braces, _ = text.rpartition('}}')
      text += braces
      print(text)
      break
    print(text)


