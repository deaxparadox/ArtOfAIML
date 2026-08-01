# Tool Calling

See [Agent Architecture](agent-architecture.md) for the loop this mechanism plugs into.

## What is it?

Tool calling is the mechanism by which an LLM requests that a specific function be executed, with specific arguments, instead of only producing free-form text — the model outputs a structured request (which tool, with what arguments), and the surrounding system executes it and returns the result.

Continuing [Agent Architecture](agent-architecture.md)'s loop: tool calling is specifically how the model's "decide the next action" step gets communicated in a way the surrounding program can actually act on reliably. Instead of the model writing "I should check the horoscope for Aquarius" in prose, which a human or fragile text-parsing code would have to interpret, tool calling has the model directly output something structured — a specific function name and a specific set of arguments — that the calling code can dispatch directly.

## Why does it exist?

[Agent Architecture](agent-architecture.md)'s toy example used a scripted `decide_next_action` function standing in for an LLM, explicitly noting a real agent would replace it with a model call. Tool calling exists to make that replacement work reliably: an LLM's raw text output is free-form and can vary in phrasing even when it means the same thing — parsing arbitrary text into a specific function call with specific arguments, via regex or ad hoc string matching, is fragile and breaks in ways that are hard to predict. Tool calling removes that fragility: the model is given a description of the available tools — their names, and a schema for their arguments — up front, and produces its "call this function" decision in a fixed, structured format instead of free text, one the surrounding code can parse reliably every time.

**This is [Prompt Engineering](../llms/prompt-engineering.md)'s specificity principle, applied to actions instead of descriptions.** Just as a specific, well-formatted prompt reduces ambiguity in a text response, well-defined tool schemas — name, description, argument types — reduce ambiguity in an action request. Providing too many tools, or tools with unclear or overlapping purposes, reintroduces the exact ambiguity structured calling was meant to remove: the model can still choose the wrong tool, or fill in an argument incorrectly, even with a rigid output format, if the tool's own description doesn't make its purpose and correct use clear enough.

## How does it work?

Tools are described to the model up front with a name, a natural-language description, and a schema for their arguments. Instead of directly producing a final answer, the model can produce a structured tool call matching one of those schemas. The surrounding program — never the model itself — executes the requested function with those arguments, and the result is fed back into the conversation as an observation, closing the loop from [Agent Architecture](agent-architecture.md).

## Example

The tool-call format below matches OpenAI's documented API shape. What follows is a representative example of that format, not a claim about what any specific model decided to output — the dispatch logic, taking a tool call and actually running the corresponding function, is real and verified:

```python
import json

def get_horoscope(sign):
    return f"{sign}: Next Tuesday you will befriend a baby otter."

TOOLS = {"get_horoscope": get_horoscope}

# A representative tool call in the documented response format
tool_call = {
    "type": "function_call",
    "call_id": "call_2345abc",
    "name": "get_horoscope",
    "arguments": '{"sign": "Aquarius"}',
}

fn = TOOLS[tool_call["name"]]
args = json.loads(tool_call["arguments"])
result = fn(**args)
print(result)  # -> Aquarius: Next Tuesday you will befriend a baby otter.
```

Note what's real here and what isn't: the tool-call structure — `name`, `call_id`, and a JSON-encoded `arguments` string — is the actual, documented shape a model's structured output takes. Whether a real model would have chosen to call `get_horoscope` with exactly this argument for a given prompt isn't something this example verifies; what it verifies is the mechanical part that matters for building the system — parsing that structure and dispatching it to the right function with the right arguments, reliably, every time.

## Where is it used?

Every agent that needs to take real actions rather than just produce text — looking something up, calling an internal API, running a calculation — and any system where an LLM's output needs to trigger code rather than just be read.

## Advantages

- **Removes the fragility of parsing free-form text into a specific action**, replacing it with a fixed, reliable structure.
- **Lets the model choose from a well-defined set of available actions**, rather than needing to be told exactly how to phrase every possible request.
- **Execution stays entirely in the surrounding system's control** — the model requests an action, but never runs it directly.

## Limitations

- **A model can still call the wrong tool, or fill in the wrong arguments**, even with a rigid structured format — the format guarantees the shape of the request is parseable, not that its content is correct.
- **Vague, overlapping, or too-numerous tool descriptions reintroduce the ambiguity** structured calling was meant to remove.
- **Every tool call is a place a model's mistake becomes a real action**, not just an incorrect sentence — the structure makes execution reliable, it doesn't make the underlying decision correct.

## Production considerations

- **Arguments arrive as untrusted input from the model**, exactly like user input from an external source — they should be validated before being used to run a real action, not trusted just because they came through a structured format.
- **A tool's own description is part of the system's real behavior.** A vague or misleading description can cause a technically well-formed, structurally valid tool call to still be the wrong one.
- **Tool execution needs the same error handling as any other code path that can fail.** A failed tool call needs to become a clear observation the model, or a human, can act on — not a silent failure that leaves the agent stuck.

## Common mistakes

- **Trusting tool call arguments as if they were already validated**, simply because they arrived in a structured format rather than free text.
- **Writing vague or overlapping tool descriptions**, then being surprised when the model picks the wrong one.
- **Not handling a failed tool execution explicitly**, leaving an agent with no way to recover or report the failure back into its own loop.

## Interview questions

### Basic

- What problem does tool calling solve that free-text parsing doesn't?
- Who actually executes the requested function: the model, or the surrounding system?

### Intermediate

- Why can a model still make a mistake with tool calling, even though the format itself is structured and reliable?
- Why do vague or overlapping tool descriptions cause real problems, even with a strict output schema?

### Advanced

- Why should tool call arguments be validated the same way external user input would be, even though they came from a model rather than a user?
- Design a tool description for an action with real consequences (e.g. sending an email). What would make that description clear enough to reduce the chance of it being called incorrectly?
