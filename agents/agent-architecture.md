# Agent Architecture

## What is it?

An agent, in the LLM sense, is a system that uses a language model not just to generate a single response, but to decide what to do next in a loop — observing the current state, choosing an action (which might mean calling a tool, searching, or asking a follow-up), and repeating until the task is done.

If a plain LLM call is like asking someone a single question and getting one answer, an agent is like giving that same person a task, a set of tools, and letting them work through it step by step — checking their own progress and adjusting their next move based on what they observe, rather than answering everything in one shot from what they already know.

## Why does it exist?

[Prompt Engineering](../llms/prompt-engineering.md) and [RAG](../rag/what-is-rag.md) both operate within a single request/response cycle: one prompt in, one answer out, even when that prompt includes retrieved context. Many real tasks don't fit that shape. Answering "what's the current weather in Tokyo, and should I bring an umbrella" requires actually calling a weather API, not generating plausible-sounding text. A multi-step task — "book me a flight, then email me the itinerary" — requires several dependent actions, where the right next step depends on what happened after the previous one. Agent architecture exists to let an LLM's output steer a loop of actions and observations, instead of being the final output itself.

**When a task needs an agent, versus a single prompt or RAG call:** a single prompt (with or without retrieval) is right when the task is answerable from one relevant piece of context and doesn't require any external side effect — no live lookup, no action outside the model's own text output. An agent is worth its added complexity specifically when the task requires taking real actions and adapting based on their results, or requires multiple dependent steps where each step's outcome changes what to do next — not by default for every task, since an agent loop is slower, harder to debug, and more expensive (multiple model calls instead of one) than a single request/response.

## How does it work?

The standard agent loop:

1. **Observe** the current state — the task, plus whatever has happened so far.
2. The LLM **decides the next action** — often expressed as calling a specific tool with specific arguments (covered in depth in the next chapter, Tool Calling).
3. The action is **executed outside the model itself** — actually calling an API, running a query — and its result becomes a new observation.
4. **Repeat** from step 1 with the updated state, until the model decides the task is complete.

This loop is what lets an agent handle a task whose full solution path isn't known in advance — each step's result can change what the next step should be.

## Example

A real LLM call isn't available to verify here, so this demonstrates the actual control-flow structure honestly: a scripted decision function stands in for the LLM's role — deciding the next action from the current state — while the loop, tools, and state updates are real and fully verifiable. A toy task with a genuine multi-step dependency: summing the prime numbers below 20, one number at a time, without knowing in advance how many steps that takes.

```python
def add(a, b):
    return a + b

def is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True

TOOLS = {"is_prime": is_prime, "add": add}

def decide_next_action(state):
    # Stands in for an LLM call: given the current state, pick the next
    # tool and arguments. Scripted here so the example is deterministic
    # and verifiable, but this is exactly the decision point a real agent
    # delegates to a language model.
    if state["n"] >= 20:
        return ("finish", {})
    if state.get("checked") is None:
        return ("is_prime", {"n": state["n"]})
    if state["checked"]:
        return ("add", {"a": state["sum"], "b": state["n"]})
    return ("advance", {})

state = {"n": 2, "sum": 0, "checked": None}

while True:
    action, args = decide_next_action(state)
    if action == "finish":
        break
    elif action == "is_prime":
        state["checked"] = TOOLS["is_prime"](**args)
    elif action == "add":
        state["sum"] = TOOLS["add"](**args)
        state["n"] += 1
        state["checked"] = None
    elif action == "advance":
        state["n"] += 1
        state["checked"] = None

print(state["sum"])  # -> 77
```

The loop runs until the model (here, the scripted stand-in) decides the task is finished — it doesn't know in advance how many numbers it'll need to check, only that it should keep observing and acting until `n` reaches 20. The final sum, 77, is the verified sum of the primes below 20 (2+3+5+7+11+13+17+19). A real agent replaces `decide_next_action` with an LLM call that reads the current state and chooses a tool — everything else about the loop's structure stays the same.

## Where is it used?

Coding assistants that read a file, run a command, and decide what to do next based on the output; customer support systems that look up an order, check a policy, and take an action; any task where the right sequence of steps can't be fully planned out in a single prompt ahead of time.

## Advantages

- **Handles tasks whose full solution path isn't known in advance**, adapting the next step based on what actually happened after the previous one.
- **Can take real actions**, not just generate text — calling APIs, running code, querying a database.
- **Decomposes a large task into smaller, individually simpler decisions**, rather than requiring one perfect answer in a single shot.

## Limitations

- **Every step in the loop is a separate model call**, multiplying latency and cost compared to a single prompt.
- **Errors can compound across steps.** A wrong decision early in the loop changes the state every subsequent step reasons from, unlike a single-shot prompt where a mistake is isolated to one response.
- **The loop needs an exit condition that actually works.** A model that never decides it's "done" — or decides too early — produces a run that either loops indefinitely or stops short of the task.

## Production considerations

- **Runaway loops are a real operational risk**, not a theoretical one — a production agent needs a hard step limit or timeout independent of the model's own judgment about when to stop.
- **Each tool call is a place a real-world side effect can happen** — sending an email, charging a payment — which means an agent's actions need the same review and safeguards as any other code path that takes real actions, not just a "text output" level of trust.
- **Cost scales with the number of steps a task takes**, which is harder to predict up front than the cost of a single prompt — worth monitoring per-task cost directly, not just per-request cost.

## Common mistakes

- **Giving an agent open-ended tasks with no step limit**, risking a loop that runs far longer, and costs far more, than intended.
- **Treating agent actions as equivalent in risk to plain text generation**, when a tool call can have a real, irreversible side effect a bad text response never could.
- **Debugging an agent by only looking at its final answer**, rather than the full sequence of observations and actions that led there — the same reasoning [Evaluation](../rag/evaluation.md) already applied to separating retrieval from generation applies here, one level up: know which step in the loop actually went wrong.

## Interview questions

### Basic

- What is the core loop an agent runs, and how does it differ from a single prompt-response call?
- Why might a task need an agent instead of a single RAG-augmented prompt?

### Intermediate

- Why does an agent's cost and latency scale differently than a single LLM call's?
- Why is a hard step limit necessary for a production agent, even though the model is expected to decide when it's done?

### Advanced

- An agent takes a harmful or incorrect action partway through a multi-step task. How would you design the system to limit the damage from a single wrong decision?
- Why can errors compound across an agent's steps in a way they don't in a single-shot prompt, and what does that imply about how you'd debug a bad outcome?
