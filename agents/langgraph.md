# LangGraph

See [Agent Architecture](agent-architecture.md) for the observe/decide/act loop this chapter expresses as an explicit graph.

## What is it?

LangGraph is a framework for building agents as an explicit graph of nodes and edges — each node a step (often an LLM call or a tool call), each edge a transition to the next step, with the graph structure itself defining how the agent loop from the previous chapter actually gets wired together and executed.

If [Agent Architecture](agent-architecture.md)'s example expressed the observe/decide/act loop as a plain Python `while` loop with an if/elif dispatch, LangGraph expresses the same structure as an explicit graph: nodes are the steps, and edges — including conditional ones — are the decision logic for what happens next, made into a first-class, inspectable part of the program instead of buried inside a loop's control flow.

## Why does it exist?

[Agent Architecture](agent-architecture.md) showed the loop working as plain Python control flow, which works fine for a small example, but that structure gets hard to reason about, test, and modify as the number of possible steps and branches grows: a single sprawling function with deeply nested if/elif logic doesn't clearly show what states the agent can be in or how it can move between them. LangGraph exists to make that structure explicit: the graph itself is a declared, inspectable object — nodes and edges you can list, visualize, and modify independently — rather than control flow embedded inside a function body.

**When plain control flow is enough, versus when a framework like LangGraph earns its keep:** a simple, linear agent loop with one or two decision points is easy enough to hand-write directly, and adding a framework's structure and dependencies is unnecessary overhead. LangGraph's explicit graph structure earns its complexity specifically once an agent has multiple branching paths, needs to loop back to earlier steps conditionally, or needs the built-in features the framework provides on top of the raw loop — state persistence across steps, resuming a paused run, visualizing the flow — capabilities that would otherwise need to be hand-built and maintained separately.

## How does it work?

- Define a **State** (typically a `TypedDict`) describing what data flows through the graph.
- Define **nodes**: plain functions that take the current state and return updates to it.
- Connect nodes with **edges**: fixed edges always go to the same next node; **conditional edges** run a function against the current state to decide which node to go to next — exactly how a loop with a termination condition gets expressed, routing back to a previous node or forward to a special `END`, based on the state.
- **Compile** the graph into a runnable object, then **invoke** it with an initial state.

## Example

Reimplementing the exact task from [Agent Architecture](agent-architecture.md) — sum the primes below 20 — using LangGraph's actual API this time, not a hand-written loop:

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    n: int
    total: int

def check_and_add(state: State) -> State:
    n = state["n"]
    is_prime = n >= 2 and all(n % d != 0 for d in range(2, int(n**0.5) + 1))
    new_total = state["total"] + n if is_prime else state["total"]
    return {"n": n + 1, "total": new_total}

def route(state: State) -> Literal["check_and_add", "__end__"]:
    return END if state["n"] >= 20 else "check_and_add"

builder = StateGraph(State)
builder.add_node(check_and_add)
builder.add_edge(START, "check_and_add")
builder.add_conditional_edges("check_and_add", route)
graph = builder.compile()

result = graph.invoke({"n": 2, "total": 0})
print(result)  # -> {'n': 20, 'total': 77}
```

Same task, same verified answer (77) as [Agent Architecture](agent-architecture.md)'s hand-written loop — but here, the looping behavior is expressed declaratively: `route` is a conditional edge that sends execution back to `check_and_add` or forward to `END`, rather than an if/elif branch buried inside a while loop's body. In a real agent, `check_and_add` would typically be an LLM call deciding the next action, and `route` would inspect what the model decided; the graph structure around them stays the same either way.

## Where is it used?

Building agents with multiple steps, branches, or loops that need to be inspected, tested, or modified as their own structure — customer support flows with several possible paths, multi-step research or coding agents, any agent complex enough that hand-written control flow becomes hard to follow.

## Advantages

- **Makes an agent's possible steps and transitions an explicit, inspectable structure**, rather than logic buried inside a function body.
- **Provides built-in support for state persistence, resuming a paused run, and visualizing the graph**, which would otherwise need to be built and maintained by hand.
- **Conditional edges give a clean, declared way to express branching and looping**, directly visible as part of the graph rather than nested control flow.

## Limitations

- **Adds real structure and a dependency for tasks simple enough that a hand-written loop** — like [Agent Architecture](agent-architecture.md)'s example — would have done the job with less code and no new concepts to learn.
- **The graph itself doesn't remove the underlying agent risks already covered** — runaway loops, compounding errors, risky tool calls. It only changes how the control flow around them is expressed.
- **Debugging a complex graph still requires understanding what state each node received and returned.** A graph with many branches can become as hard to reason about as a complex function, just in a different shape.

## Production considerations

- **The same step-limit and timeout safeguards from [Agent Architecture](agent-architecture.md) still apply.** A conditional edge that never routes to `END` is exactly as much a runaway-loop risk as a while loop that never breaks.
- **State persistence features are valuable for long-running or resumable agents**, but they mean the graph's state needs the same versioning and reproducibility discipline as any other production data, per [ML Workflow](../machine-learning/ml-workflow.md).
- **Visualizing and testing individual nodes in isolation is one of the framework's real practical benefits**, since a node is just a function — but that benefit is only realized if nodes are actually kept testable in isolation, rather than reaching into shared, hidden state.

## Common mistakes

- **Reaching for a graph framework for a task simple enough that a plain function or loop would be clearer**, with fewer moving parts.
- **Writing a conditional edge that can't actually reach `END` under some state**, recreating the runaway-loop risk the framework doesn't automatically prevent.
- **Assuming the graph structure alone makes an agent safe or correct**, when it only makes the control flow explicit — the underlying decisions and actions still need the same scrutiny as any agent's.

## Interview questions

### Basic

- What are the two basic building blocks of a LangGraph graph?
- How does a conditional edge differ from a regular edge?

### Intermediate

- Why might a graph-based structure be easier to reason about than an equivalent hand-written loop, once an agent has several branches?
- What does compiling a graph actually do, before it can be invoked?

### Advanced

- How would you express the same loop-with-termination-condition pattern from [Agent Architecture](agent-architecture.md) using LangGraph's conditional edges, and what changes about how that logic is expressed?
- What real risks from [Agent Architecture](agent-architecture.md) — runaway loops, compounding errors — does a graph framework not automatically solve, just by using it?
