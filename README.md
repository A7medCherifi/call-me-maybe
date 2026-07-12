*This project has been created as part of the 42 curriculum by acherifi.*

# call-me-maybe

## Description
`call-me-maybe` is a Python project about function calling with LLMs (Qwen3-0.6B).
The goal is to convert a natural language prompt into a structured function call.
The project focuses on constrained decoding, so the output format stays valid and predictable with clean function name and arguments.

---

## Instructions

### Requirements
- `Python 3.10+ (or newer)`
- `uv`

### Installation

Install dependencies (create a virtual environment Recommended):
```bash
make install
```

### Execution
Run the program (example):
```bash
make run
```

---

## Algorithm explanation (Constrained Decoding)
for the constrained decoding approach i mask the LLM's vocabulary accordingly:

- Numbers & Integers: i mask the LLM's vocab. And only allowed to output digits, decimals, signs, or completion tokens (like commas).

- Booleans: The vocabulary is strictly reduced to the tokens for "true" or "false".

- Strings: The LLM is allowed to generate freely, but the algorithm actively monitors for the closing quotation mark (") to know exactly when the string is finished.

This reduces invalid outputs and makes integration with tools safer.

---

## Design decisions
Main implementation choices:

#### Stage 1: Forcing a Valid Function Name
First, the LLM must choose which function to call. i restrict its vocabulary so it can only output tokens that match the predefined function names provided in my initial data. If the model tries to hallucinate a function that doesn't exist, the algorithm blocks that token and forces it to pick a valid one.

#### Stage 2: Injecting the JSON Scaffolding
Once the valid function name is selected, i don't even wait for the LLM to format the JSON. i step in and i use a concept that called Token injection by manually inject the structural tokens like: `"parameters": {` directly into the input/output stream. This guarantees perfect syntax without wasting compute.

#### Stage 3: Injecting Parameter Keys
Because i already know which function was selected in Stage 1, i know exactly what parameters are required. The algorithm automatically injects the exact parameter keys (e.g., "name": ) into the prompt. The LLM's only job is to provide the values.

#### Stage 4: Type-Constrained Values (The Magic)
This is where the dynamic constraint happens. i look at the expected data type of the current parameter and mask the LLM's vocabulary, for more details its above in Algorithm explanation.

#### Stage 5: Safe Closure
Once all required parameters have been successfully generated and formatted, the algorithm forcefully injects the closing brackets (}}) to terminate the generation. The final output is guaranteed to be fully parseable JSON, ready to be executed by your application honey, ta wahd may9ra hadchi wa9ila lol.

---

## Performance analysis
**Accuracy (100% Guaranteed Syntax):**\
The algorithm physically blocks the model from hallucinating function names or generating invalid JSON. By masking invalid tokens, the structural and type accuracy is absolute.

**Speed (Highly Efficient):**\
I save compute time by automatically injecting the JSON structure tokens (like "prompt": "", name: "", "parameters": {). The AI only spends time generating the actual values, making it significantly faster than waiting for standard text generation.

**Reliability:**\
It completely eliminates the most common LLM failure modes—like missing brackets, trailing commas, or outputting text when a number is required. Downstream tools will always receive perfectly parseable JSON.

---

## Challenges faced
Some difficulties and solutions:

**Perfect prompts**  
i spend a lot of time looking for the best prompt that can let the llm generate a valid values and also dont take much time in encoding.

**speeding up the llm generation**  
at first it was so challenging to make the llm generation fast so it tok me time to optimize it like that.

---

## Testing strategy
To guarantee that the constrained decoding engine works flawlessly, the implementation was validated against both structural and edge-case scenarios. Since the core promise of this project is 100% valid JSON, the testing strategy heavily focuses on breaking the generation loop.

Here is how the system is validated:

**The `json.loads()` Guarantee:**\
The ultimate test for every generated output is passing it directly into Python's native `json.loads()` function. If the output throws a parsing error (e.g., missing quotes, trailing commas, unclosed brackets), the test fails.

**Type Boundary Testing:**\
We specifically test parameters with strict types (`number`, `boolean`, `integer`). For example, we prompt the model in a way that encourages it to generate a string (e.g., "Set the boolean to 'maybe'"), and verify that the logit mask successfully forces it into a valid `true` or `false` token instead.

**Vocabulary Verification:**\
Unit tests validate the internal `__get_valid_digits` and `__get_valid_boolean` methods to ensure no illegal characters (like letters in a number field) slip into the allowed token lists.

**Edge-Case Escaping:**\
Evaluated the engine’s ability to handle nested quotes or complex strings (e.g., \") without prematurely triggering the end-of-string token logic.

---

## Example usage

### Example 1
Input:
```json
{
    "prompt": "What is the sum of 265 and 345?"
}
```

Expected structured output (example):
```json
{
    "prompt": "What is the sum of 265 and 345?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 265.0,
        "b": 345.0
    }
}
```

### Example 2
Input:
```json
{
    "prompt": "What is the square root of 16?"
}
```

Expected structured output (example):
```json
{
    "prompt": "What is the square root of 16?",
    "name": "fn_get_square_root",
    "parameters": {
        "a": 16.0
    }
}
```

---

## Resources
Classic references:

- [What is LLM](https://aws.amazon.com/what-is/large-language-model/)
- [How LLMs Work](https://blog.langformers.com/how-llms-work/)
- [LLM Embedding Explained](https://huggingface.co/spaces/hesamation/primer-llm-embedding?section=references)
- [Understanding NLP Architecture](https://medium.com/@bipasha_bg/understanding-nlp-architecture-3b0a50c8192f)
- [Constrained Decoding](https://zeroentropy.dev/concepts/constrained-decoding/#what-gets-constrained)
- [JSON Decoding for Local LLMs](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/)
- [Everything about LLMs in simple](https://algomaster.io/learn/ai-engineering/what-are-llms)

### AI usage
- Help me with creating this README file
- Explain many concepts in more details and examples
- Helped me in code enhancement
- Provides me with usefull resources
