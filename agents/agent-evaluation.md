# Agent Evaluation

See [Evaluation](../rag/evaluation.md) for the retrieval-vs-generation split this chapter extends across an entire agent trajectory, and [Agent Architecture](agent-architecture.md) for the loop being evaluated.

## What is it?

Agent evaluation is the practice of measuring how well an agent performs across an entire multi-step trajectory — not just whether its final answer is correct, but whether each intermediate decision (which tool to call, whether to retry, when to stop) was actually a good one along the way.

[Evaluation](../rag/evaluation.md) separated retrieval quality from generation quality specifically because a single end-to-end score doesn't say which half to fix. Agent evaluation extends that same idea across an entire sequence of steps: a wrong final answer could come from a bad first decision, a wrong tool call, a failure to retry when it should have, or — as this chapter's example shows directly — a correct process that's still meaningfully worse than it needed to be.

## Why does it exist?

[LLM Evaluation / Benchmarks](../llms/llm-evaluation-benchmarks.md) already established that evaluating a single LLM output is harder than a classical accuracy score, since the output is open-ended. Agent evaluation goes a step further: an agent produces not just one output, but a whole sequence of decisions and actions, per [Agent Architecture](agent-architecture.md)'s own loop, and a single final-answer score conflates all of them into one number. If an agent reaches the right final answer through a wasteful or risky process, evaluating only the final answer calls that a success — masking a real, latent problem that could cause a worse outcome next time, on a slightly different input.

**Outcome-based vs. trajectory-based evaluation — a real choice, the same split [Evaluation](../rag/evaluation.md) already drew for RAG.** Outcome-based evaluation only checks the final result — simpler to measure, but blind to how the agent got there. Trajectory-based evaluation checks the actual sequence of steps — did it call the right tools, in a reasonable order, without unnecessary or risky actions — catching process problems outcome-based evaluation misses entirely, at the cost of being harder to define and usually requiring either a human reviewer or a carefully designed automated check for each step.

## How does it work?

**Outcome evaluation** compares the agent's final result against a known correct answer, the same style already covered for RAG and LLMs generally. **Trajectory evaluation** checks the sequence of tool calls and decisions against what a reasonable trajectory should look like — unnecessary tool calls, a missing necessary one, stopping too early or looping too long, per [Agent Architecture](agent-architecture.md#production-considerations)'s own step-limit concern. **Step-level checks** verify each individual tool call's arguments and results were reasonable, related to but distinct from [Guardrails](guardrails.md) — evaluation checks after the fact whether things went well, while a guardrail blocks before the fact.

## Example

Two agents reaching the exact same correct outcome — the sum-of-primes-below-20 task from [Agent Architecture](agent-architecture.md) — through meaningfully different trajectories:

```python
def is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True

def run_efficient_agent(limit=20):
    total, tool_calls = 0, 0
    for n in range(2, limit):
        tool_calls += 1
        if is_prime(n):
            total += n
    return total, tool_calls

def run_wasteful_agent(limit=20):
    total, tool_calls = 0, 0
    for n in range(2, limit):
        for _ in range(3):  # redundantly re-checks the same thing 3 times
            tool_calls += 1
            result = is_prime(n)
        if result:
            total += n
    return total, tool_calls

efficient_outcome, efficient_calls = run_efficient_agent()
wasteful_outcome, wasteful_calls = run_wasteful_agent()
print(f"efficient agent -> outcome: {efficient_outcome}, tool_calls: {efficient_calls}")
print(f"wasteful agent  -> outcome: {wasteful_outcome}, tool_calls: {wasteful_calls}")
```

```text
efficient agent -> outcome: 77, tool_calls: 18
wasteful agent  -> outcome: 77, tool_calls: 54
```

Both agents reach the identical, correct answer — 77. Outcome-only evaluation scores them exactly the same: both correct, indistinguishable. Trajectory evaluation reveals what outcome evaluation can't see at all: the wasteful agent used exactly three times as many tool calls to reach the same result, a real efficiency problem — and in a system with real-world tool calls instead of a pure calculation, three times the actions is also three times the exposure to something going wrong.

## Where is it used?

Comparing candidate agent designs or prompt changes before deployment, monitoring whether a production agent's behavior stays efficient and safe over time, and diagnosing why an agent's final answers are wrong when the answer alone doesn't reveal which step failed.

## Advantages

- **Reveals inefficiency and risk that outcome-only evaluation hides entirely**, as this chapter's example shows directly — identical correctness, a threefold difference in actual cost.
- **Pinpoints which step in a multi-step process actually failed**, rather than only knowing the final answer was wrong.
- **Extends naturally to safety evaluation** — checking not just whether an agent reached a good outcome, but whether it took any risky or unnecessary actions on the way.

## Limitations

- **Defining a "reasonable" trajectory is harder than defining a correct outcome.** Whether an extra tool call was actually wasteful, versus reasonably cautious, isn't always obvious.
- **Trajectory evaluation usually requires more manual design or human review** than a straightforward outcome comparison, the same added cost [Evaluation](../rag/evaluation.md) already noted for building any real test set.
- **A good trajectory doesn't guarantee a good outcome, and vice versa** — both need to be checked, neither alone is sufficient.

## Production considerations

- **Trajectory metrics (tool call count, step count, retries) are cheap to log and monitor continuously**, unlike full human review of every trajectory, and can catch a growing inefficiency problem before it becomes a cost or safety issue.
- **A sudden increase in average tool calls per task is a real signal worth alerting on**, the same way a latency or error-rate spike would be, per [Observability](../mlops/observability.md).
- **Evaluation should run on every meaningful change to an agent's prompts, tools, or model version**, the same discipline already established in [ML Workflow](../machine-learning/ml-workflow.md) and [Evaluation](../rag/evaluation.md).

## Common mistakes

- **Evaluating an agent only on final-answer correctness**, missing exactly the kind of inefficiency this chapter's example demonstrates — a threefold cost difference invisible to outcome-only checks.
- **Treating a correct outcome as proof the process was sound**, when a lucky or wasteful path can produce the same right answer as an efficient, well-reasoned one.
- **Not monitoring trajectory metrics in production**, missing a gradual efficiency regression until it shows up as a cost or latency problem instead.

## Interview questions

### Basic

- What's the difference between outcome-based and trajectory-based agent evaluation?
- Why did both agents in this chapter's example score identically under outcome-only evaluation?

### Intermediate

- Why can a correct final answer still represent a real problem in an agent's process?
- What trajectory metrics would you monitor continuously in production, and why are they cheaper to track than full trajectory review?

### Advanced

- Design an evaluation approach for an agent that must be both efficient (few unnecessary tool calls) and safe (no risky actions), given that trajectory quality is harder to define than outcome correctness.
- A production agent's tool-call count per task has crept up over several weeks with no corresponding change in correctness. How would you investigate and what would you check first?
