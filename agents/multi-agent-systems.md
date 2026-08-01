# Multi-Agent Systems

See [Agent Architecture](agent-architecture.md) and [LangGraph](langgraph.md) for the single-agent loop and graph structure this chapter extends to multiple agents.

## What is it?

A multi-agent system is a setup where multiple LLM-driven agents, each with their own role, tools, or scope, work together on a task — communicating through structured state and dividing responsibility, rather than one single agent handling everything alone.

If a single agent is one person working through a task with a set of tools, a multi-agent system is a small team: an orchestrator who breaks the task down and delegates, and specialists who each handle their own piece and report back, rather than one generalist trying to do everything themselves.

## Why does it exist?

[Agent Architecture](agent-architecture.md) and [Tool Calling](tool-calling.md) both describe a single agent with one running state and one set of available tools. As a task grows — more tools to choose between, more accumulated history to reason over — a single agent's context grows with it, and so does the chance its one decision loop picks the wrong action. Multi-agent systems exist to manage that growth the same way a large software project gets decomposed into modules: give each agent a narrower job, a smaller relevant toolset, and a smaller amount of context to reason over, rather than one agent holding an entire task in its single context window and decision loop at once.

**Single agent vs. multi-agent is a real trade-off, not a strictly-better upgrade.** A single agent is simpler to build, debug, and reason about — one loop, one state, one place to look when something goes wrong. Multi-agent systems earn their added complexity specifically when a task naturally decomposes into distinct roles or specialties that benefit from focused context and tools, or when a single agent's context would otherwise become too large or too broad to reason over reliably. Splitting a task into multiple agents purely because it seems more sophisticated, when a single well-scoped agent would do the job, adds coordination overhead — miscommunication, redundant work, unclear ownership of a decision — without a matching benefit.

## How does it work?

A common pattern: an orchestrator receives the overall task, breaks it into sub-tasks, and delegates each to a specialized sub-agent, potentially with its own tools, prompt, and scope. Sub-agents return their results to the orchestrator, which combines them — and may loop back, asking a sub-agent to redo something, or delegating a new sub-task based on what it learned, before producing a final result. This is the same observe/decide/act loop from [Agent Architecture](agent-architecture.md), just with the state now including other agents' outputs, and the action sometimes being "delegate to another agent" instead of "call a tool directly." Communication between agents typically happens through structured state, not unlike the tool-call structure from the previous chapter — one agent's output becomes another's input in a defined format, rather than an unconstrained conversation.

## Example

This is literally a bigger version of the graph from [LangGraph](langgraph.md) — an orchestrator node and two specialist nodes, verified and running:

```python
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    n: int
    is_prime: Optional[bool]
    is_even: Optional[bool]
    report: Optional[str]

def prime_specialist(state: State) -> State:
    n = state["n"]
    result = n >= 2 and all(n % d != 0 for d in range(2, int(n**0.5) + 1))
    return {"is_prime": result}

def parity_specialist(state: State) -> State:
    return {"is_even": state["n"] % 2 == 0}

def orchestrator_combine(state: State) -> State:
    report = f"{state['n']} is {'prime' if state['is_prime'] else 'not prime'} and {'even' if state['is_even'] else 'odd'}"
    return {"report": report}

builder = StateGraph(State)
builder.add_node(prime_specialist)
builder.add_node(parity_specialist)
builder.add_node(orchestrator_combine)

builder.add_edge(START, "prime_specialist")
builder.add_edge(START, "parity_specialist")
builder.add_edge("prime_specialist", "orchestrator_combine")
builder.add_edge("parity_specialist", "orchestrator_combine")
builder.add_edge("orchestrator_combine", END)

graph = builder.compile()
result = graph.invoke({"n": 17, "is_prime": None, "is_even": None, "report": None})
print(result["report"])  # -> 17 is prime and odd
```

Two specialists (`prime_specialist`, `parity_specialist`) each work on their own narrow question, independently, and an orchestrator (`orchestrator_combine`) combines their outputs into a final report — verified correct: 17 is indeed prime and odd. In a real multi-agent system, each specialist node would typically be its own LLM call with its own prompt and tools, not a plain function — but the coordination structure, delegate out, combine back, is exactly this graph shape, just with more sophisticated nodes.

## Where is it used?

Complex research or analysis tasks split across specialized sub-agents, coding assistants that separate planning from execution from review, and any system where a single agent's context or toolset would otherwise become too broad to reason over reliably.

## Advantages

- **Keeps each agent's context and toolset narrow and focused**, rather than one agent needing to reason over everything at once.
- **Mirrors a familiar, well-understood decomposition pattern** — breaking a large task into smaller, owned pieces — already used throughout software design.
- **Specialist agents can be developed, tested, and improved somewhat independently**, similar to how independent modules in a larger system can be.

## Limitations

- **Coordination itself becomes a real cost.** Agents miscommunicating, redundant work between specialists, or unclear ownership of a decision are new failure modes a single agent doesn't have.
- **Every hand-off is another place information can be lost, misinterpreted, or subtly changed**, compared to a single agent with direct access to its own full history.
- **More agents means more total model calls for the same task**, directly increasing latency and cost — the same trade-off already raised for a single agent's step count in [Agent Architecture](agent-architecture.md).

## Production considerations

- **Multi-agent systems inherit every risk already covered for a single agent, per agent** — runaway loops, compounding errors, risky tool calls — multiplied across however many agents are involved.
- **Debugging means tracing which agent made which decision and what it actually received from the others**, not just reading one agent's history — a real increase in operational complexity over a single agent.
- **Cost and latency scale with the number of agents and hand-offs involved**, which is harder to estimate up front than a single agent's step count, and worth monitoring per-task, not just per-agent.

## Common mistakes

- **Splitting a task into multiple agents by default**, when a single, well-scoped agent with clear tools would have done the job with far less coordination overhead.
- **Assuming structured hand-offs between agents eliminate miscommunication entirely**, rather than just making it more visible and debuggable than free-form back-and-forth would be.
- **Not tracing which specific agent and step introduced an error**, and instead treating the whole multi-agent system as one opaque unit when something goes wrong.

## Interview questions

### Basic

- What problem does splitting a task across multiple agents solve that a single agent doesn't?
- What's the role of an orchestrator in a multi-agent system?

### Intermediate

- Why does a multi-agent system typically cost more and run slower than an equivalent single-agent solution?
- What new failure modes does coordination between agents introduce that a single agent doesn't have?

### Advanced

- A multi-agent system produces a wrong final result. How would you determine which agent, and which hand-off, is actually responsible?
- When would a single, well-scoped agent be the better choice over decomposing a task into multiple agents, even if the task technically has distinct sub-parts?
