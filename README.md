*This project has been created as part of the 42 curriculum by acherifi.*

# call-me-maybe

## Description
`call-me-maybe` is a small Python project about function calling with LLMs.

The goal is to convert a natural language prompt into a structured function call.
The project focuses on constrained decoding, so the output format stays valid and predictable.

In simple words: the user writes normal text, and the program returns a clean function name with arguments.

---

## Instructions

### Requirements
- Python 3.10+ (or newer)
- `pip`

### Installation
Clone the repository:
```bash
cd call-me-maybe
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Execution
Run the program (example):
```bash
python3 main.py --prompt "Book a table for 2 people at 8pm"
```

If your entry file is different, replace `main.py` with your actual file.

---

## Algorithm explanation (Constrained Decoding)
The constrained decoding approach is:

1. Define an allowed schema for function calls:
   - function name
   - argument names
   - argument types

2. Read the user prompt.

3. Generate candidate output tokens step by step, but allow only tokens that keep the output valid according to the schema.

4. Reject invalid choices during generation (wrong function name, wrong key, broken JSON structure, wrong type, etc.).

5. Return the final structured function call only when it fully matches the schema.

This reduces invalid outputs and makes integration with tools safer.

---

## Design decisions
Main implementation choices:

- **Python-first implementation** for clarity and fast development.
- **Schema-driven validation** to enforce structure.
- **Small and modular code** (parsing, decoding constraints, validation separated).
- **Readable output format** (JSON-like structured call) to simplify debugging and testing.
- **Minimal dependencies** to keep setup simple.

---

## Performance analysis
This solution is small and designed for reliability over complexity.

- **Accuracy**: good when prompts match supported functions and argument patterns.
- **Speed**: fast for small prompts and small function schemas.
- **Reliability**: higher than free-form generation because invalid tokens are blocked during decoding.

Limits:
- If the prompt is ambiguous, the selected function may still be imperfect.
- Large schemas can make decoding slower.

---

## Challenges faced
Some difficulties and solutions:

- **Ambiguous user prompts**  
  Solved by adding stricter argument rules and clearer function descriptions.

- **Invalid intermediate outputs**  
  Solved by checking validity at each decoding step (not only at the end).

- **Type mismatch (string/int/bool)**  
  Solved by explicit type validation and conversion rules.

- **Maintaining simplicity**  
  Solved by keeping the code modular and avoiding unnecessary abstractions.

---

## Testing strategy
Validation was done with simple, practical tests:

1. **Happy path tests**  
   Prompts that clearly match one function and valid arguments.

2. **Edge case tests**  
   Missing arguments, extra arguments, wrong types, empty prompts.

3. **Negative tests**  
   Inputs that should be rejected by constraints.

4. **Regression checks**  
   Re-run previous prompts after code changes to ensure no breakage.

5. **Manual checks**  
   Human review of output readability and correctness.

---

## Example usage

### Example 1
Input:
```bash
python3 main.py --prompt "Send an email to alice@example.com with subject Meeting"
```

Expected structured output (example):
```json
{
  "function": "send_email",
  "arguments": {
    "to": "alice@example.com",
    "subject": "Meeting"
  }
}
```

### Example 2
Input:
```bash
python3 main.py --prompt "Set an alarm for 07:30 tomorrow"
```

Expected structured output (example):
```json
{
  "function": "set_alarm",
  "arguments": {
    "time": "07:30",
    "day": "tomorrow"
  }
}
```

---

## Resources
Classic references:

- OpenAI function calling / structured outputs documentation
- JSON Schema documentation: https://json-schema.org/
- Python documentation: https://docs.python.org/3/
- Articles and tutorials about constrained decoding and grammar-based generation

### How AI was used
AI was used as a support tool for:
- brainstorming the README structure,
- improving wording clarity,
- checking grammar and formatting.

AI was **not** used to replace understanding of the project logic.
Final technical choices and final implementation decisions were made manually.