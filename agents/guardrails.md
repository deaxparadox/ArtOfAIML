# Guardrails

See [Tool Calling](tool-calling.md) for the untrusted-arguments point this chapter enforces directly.

## What is it?

Guardrails are checks applied to an agent's inputs, outputs, and especially its tool calls, specifically to prevent harmful, unsafe, or unintended actions — not just checking whether a final answer looks reasonable, but checking *before* a risky action is allowed to execute.

[Tool Calling](tool-calling.md#production-considerations) already put it directly: "arguments arrive as untrusted input from the model, exactly like user input from an external source." Guardrails are the actual enforcement of that principle — a deliberate checkpoint between "the model decided to do X" and "X actually happens," the way a safety interlock on a machine stops a dangerous action even if it was requested.

## Why does it exist?

[Tool Calling](tool-calling.md) and [Agent Architecture](agent-architecture.md) both already flagged the underlying risk: "every tool call is a place a real-world side effect can happen," and arguments should be "validated before being used to run a real action." Guardrails exist to be the actual mechanism that does that validation systematically, rather than leaving it as an implicit hope that the model won't request something harmful.

**Guardrails at the tool-call layer vs. output-layer checks — a meaningfully different choice, not two names for the same thing.** Output-layer checks — scanning the model's final response for something problematic — can catch a bad response before it reaches a user, but do nothing to stop a tool call that already executed a real side effect (sent an email, charged a payment) before that check ever ran. Tool-call-layer guardrails intercept the action itself before execution — validating arguments, checking permissions, applying business rules — which is the only place a genuinely irreversible action can actually be stopped in time.

## How does it work?

**Argument validation** checks a tool call's arguments against expected types, ranges, and business rules before executing it — a direct, concrete extension of [Tool Calling](tool-calling.md)'s untrusted-input point. **Policy checks** decide whether a specific action needs different handling than a routine one — a large refund versus a small one, a record deletion versus a read. **Output scanning** checks generated text for policy violations, still useful, but not sufficient on its own for actions with real side effects, since scanning happens after the fact.

## Example

A guardrail blocking a refund that exceeds a policy threshold, before it ever executes:

```python
import json

def process_refund(order_id, amount):
    return f"Refund of ${amount:.2f} processed for order {order_id}."

TOOLS = {"process_refund": process_refund}
REFUND_APPROVAL_THRESHOLD = 500.00

def guardrail_check(tool_name, args):
    if tool_name == "process_refund" and args.get("amount", 0) > REFUND_APPROVAL_THRESHOLD:
        return False, f"Refund of ${args['amount']:.2f} exceeds the ${REFUND_APPROVAL_THRESHOLD:.2f} threshold."
    return True, None

def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = json.loads(tool_call["arguments"])
    allowed, reason = guardrail_check(name, args)
    if not allowed:
        return f"BLOCKED: {reason}"
    return TOOLS[name](**args)

print(execute_tool_call({"name": "process_refund", "arguments": json.dumps({"order_id": "A123", "amount": 45.00})}))
print(execute_tool_call({"name": "process_refund", "arguments": json.dumps({"order_id": "B456", "amount": 750.00})}))
```

```text
Refund of $45.00 processed for order A123.
BLOCKED: Refund of $750.00 exceeds the $500.00 threshold.
```

The $45 refund processes normally. The $750 refund is stopped before `process_refund` ever runs — `TOOLS[name](**args)` is never called for the blocked request. The guardrail sits between the model's decision and the actual action, exactly where [Tool Calling](tool-calling.md) said validation needed to happen, not as an afterthought checking the outcome once it's already too late to prevent.

## Where is it used?

Any agent with tools that have real financial, data, or communication consequences — processing payments or refunds, sending messages, modifying or deleting records — and any system where "the model requested it" isn't sufficient justification for an action to happen automatically.

## Advantages

- **Stops a harmful action before it happens**, not just after, as the example shows directly — the blocked refund never executes at all.
- **Encodes business rules explicitly and consistently**, rather than relying on the model to remember and apply them correctly every time.
- **Composable with other safety measures** — output scanning and human review both still have a role, guardrails just close the gap those alone leave open.

## Limitations

- **Guardrails only catch what they're explicitly built to check.** A business rule nobody thought to encode still lets an unanticipated harmful action through.
- **Overly strict guardrails block legitimate actions**, frustrating the very automation an agent is meant to provide — the threshold itself is a real trade-off, not a set-and-forget value.
- **Guardrail logic is itself code that can have bugs**, and a broken guardrail can fail open (silently allowing what it should have blocked) just as easily as it can fail closed.

## Production considerations

- **Guardrail rules need the same review and versioning discipline as any other production logic**, per [ML Workflow](../machine-learning/ml-workflow.md) — a silently outdated threshold is a real operational risk, not a static safety net.
- **A blocked action needs a real next step**, not just a dead end — which is exactly what [Human-in-the-Loop](human-in-the-loop.md) exists to provide.
- **Guardrails add a real check on every tool call**, a small but nonzero latency cost that compounds across a multi-step agent loop, per [Agent Architecture](agent-architecture.md#production-considerations).

## Common mistakes

- **Relying only on output-layer checks for actions with real side effects**, missing that the harmful action already happened before the output was ever generated.
- **Setting a guardrail threshold once and never revisiting it**, as business rules and risk tolerance change over time.
- **Assuming a guardrail that exists is a guardrail that works** — untested guardrail logic can fail silently, in either direction.

## Interview questions

### Basic

- What's the difference between a tool-call-layer guardrail and an output-layer check?
- Why did the $750 refund get blocked in this chapter's example, and at what point in execution?

### Intermediate

- Why is checking an agent's final text output not sufficient to prevent a harmful tool call?
- What does it mean for a guardrail to "fail open" versus "fail closed," and why does that distinction matter?

### Advanced

- Design a guardrail system for an agent that can send emails, process refunds, and delete customer records. What different rules would you apply to each, and why?
- How would you test that a guardrail actually blocks what it's supposed to, without accidentally blocking legitimate actions too?
