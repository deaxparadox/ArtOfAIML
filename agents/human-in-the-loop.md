# Human-in-the-Loop

See [Guardrails](guardrails.md) for the blocked action this chapter gives a real path forward.

## What is it?

Human-in-the-loop is a pattern where a high-stakes agent action pauses for explicit human approval before executing, rather than either blocking it outright or letting it proceed automatically.

[Guardrails](guardrails.md)' own example blocked a $750 refund outright, ending the story at "BLOCKED" — a dead end. Human-in-the-loop is what actually happens next in a real system: instead of simply refusing, the flagged action gets routed to a person who can review it and decide whether to approve it, turning a dead end into a real, resolvable path forward.

## Why does it exist?

[Guardrails](guardrails.md#limitations) already named the problem directly: "an overly strict guardrail blocks legitimate actions, frustrating the very automation an agent is meant to provide." A guardrail that simply refuses every refund over $500 forever, with no path to actually process a legitimate large refund, isn't a complete solution — it just moves the friction from "an unchecked action" to "a legitimate action that can never happen automatically." Human-in-the-loop exists to solve exactly this: it's not a stricter guardrail, it's a different *kind* of response to a flagged action — pause and ask, rather than block permanently or allow silently.

**Guardrail (block) vs. human-in-the-loop (pause and ask) vs. full automation (allow) — three genuinely different responses, not different strictness levels of the same thing.** Full automation is right for low-stakes, easily reversible actions where a mistake costs little. A hard block is right for actions that should never happen through this path at all, regardless of who approves it. Human-in-the-loop is right specifically for actions that are legitimate and sometimes necessary, but risky or high-stakes enough that a person should confirm the specific instance before it proceeds — the refund might be entirely legitimate, but $750 is enough money that a human should look at this particular case first.

## How does it work?

When a check flags an action as needing review rather than outright blocking or allowing it, the action is queued rather than executed or rejected immediately. A human reviewer sees the proposed action and its context — what the agent decided, why, what triggered the review — and explicitly approves or rejects it. Only after approval does the action actually execute; the agent's own loop from [Agent Architecture](agent-architecture.md) effectively pauses at that step, waiting for an external decision before continuing.

## Example

Extending [Guardrails](guardrails.md)' exact scenario: instead of a dead end, the flagged refund gets queued, then actually executes once approved:

```python
import json

def process_refund(order_id, amount):
    return f"Refund of ${amount:.2f} processed for order {order_id}."

TOOLS = {"process_refund": process_refund}
REFUND_APPROVAL_THRESHOLD = 500.00
pending_approvals = []

def guardrail_check(tool_name, args):
    if tool_name == "process_refund" and args.get("amount", 0) > REFUND_APPROVAL_THRESHOLD:
        return "needs_approval", f"Refund of ${args['amount']:.2f} exceeds the ${REFUND_APPROVAL_THRESHOLD:.2f} threshold."
    return "allowed", None

def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = json.loads(tool_call["arguments"])
    status, reason = guardrail_check(name, args)
    if status == "needs_approval":
        pending_approvals.append(tool_call)
        return f"PENDING HUMAN APPROVAL: {reason}"
    return TOOLS[name](**args)

def approve_pending(index, approved_by):
    tool_call = pending_approvals.pop(index)
    args = json.loads(tool_call["arguments"])
    result = TOOLS[tool_call["name"]](**args)
    return f"{result} (approved by {approved_by})"

call = {"name": "process_refund", "arguments": json.dumps({"order_id": "B456", "amount": 750.00})}
print(execute_tool_call(call))
print(approve_pending(0, approved_by="support_manager_jane"))
```

```text
PENDING HUMAN APPROVAL: Refund of $750.00 exceeds the $500.00 threshold.
Refund of $750.00 processed for order B456. (approved by support_manager_jane)
```

The refund isn't executed immediately, and it isn't permanently rejected either — it sits in `pending_approvals` until a human explicitly reviews it. Only once `approve_pending` is called does `process_refund` actually run. The exact same flagged action that dead-ended in [Guardrails](guardrails.md) now has a real, working resolution path: a legitimate $750 refund still gets processed, just with a human confirming this specific instance first.

## Where is it used?

High-stakes actions where a mistake is costly or hard to reverse — large financial transactions, sending communications on behalf of a company, deleting records — anywhere the right answer isn't "always allow" or "always block," but "let a person decide this specific case."

## Advantages

- **Turns a blocked action into a resolvable one**, as the example shows directly, instead of the dead end a pure guardrail leaves behind.
- **Keeps a human's judgment in the loop for exactly the cases that need it**, without requiring manual review of every routine, low-stakes action.
- **Creates a natural audit trail** — who approved what, and when — for exactly the actions where that record matters most.

## Limitations

- **Adds real latency for the specific actions it applies to.** A refund that would have processed instantly now waits for a human to become available.
- **Only as good as the humans reviewing it.** A reviewer who rubber-stamps every request without real scrutiny provides no more safety than no review at all.
- **Doesn't scale to every action.** Routing everything through human review defeats the purpose of automation — it's a targeted tool for specific high-stakes cases, not a default safety blanket.

## Production considerations

- **The approval queue itself needs monitoring** — a growing backlog of pending high-stakes actions is an operational problem, not just a safety feature working as intended.
- **Approval latency needs to be visible to whoever is waiting on the outcome**, whether that's a customer or another part of the system — a silent pending state is a poor experience even when the eventual answer is yes.
- **Who is authorized to approve which kind of action is itself a real access-control decision**, not an afterthought — the same care given to any other production permission system.

## Common mistakes

- **Routing too many actions through human review**, creating a backlog that defeats the purpose of having an agent automate the work in the first place.
- **Routing too few actions through review**, missing genuinely high-stakes cases that a guardrail alone would have let through automatically.
- **Treating approval as a formality rather than real review**, undermining the entire safety benefit the pattern is meant to provide.

## Interview questions

### Basic

- What's the difference between a guardrail blocking an action and routing it to human-in-the-loop?
- What happens to the flagged refund in this chapter's example if it's never approved?

### Intermediate

- Why doesn't full automation or an outright block fully solve the problem human-in-the-loop addresses?
- What real cost does adding a human approval step introduce, beyond just the review itself?

### Advanced

- Design the criteria for deciding which agent actions should be fully automated, which should be hard-blocked, and which should require human-in-the-loop approval.
- How would you detect that human reviewers are rubber-stamping approvals without real scrutiny, undermining the safety the pattern is meant to provide?
