# Prompt Engineering

## What is it?

Prompt engineering is the practice of designing the input given to a language model to reliably produce the output you want — choosing wording, structure, and examples that shape the model's response, without changing the model's weights at all.

If a model is a learned function, as covered in [What is Machine Learning](../machine-learning/what-is-machine-learning.md), prompt engineering is choosing the input to that function carefully. The difference from a typical function call is that the same underlying model can behave very differently depending on how the input is phrased — the "function" is enormously sensitive to its exact wording in a way a typical software function isn't.

## Why does it exist?

LLMs are trained to predict the next token given everything before it, which means the same trained model can produce very different outputs depending on how a request is framed — the model is fundamentally completing a pattern based on its input, not executing a fixed, well-defined function call the way traditional software does. Prompt engineering exists because, for a fixed model, the prompt is often the only lever available to change its behavior: no retraining, no fine-tuning, just a different framing of the same underlying request — and that framing can be the difference between a generic, unhelpful answer and a precisely useful one.

**When prompt engineering is enough, and when it isn't:** it's the cheapest, fastest lever available — no training data, no compute cost beyond the request itself, immediate iteration — and it's the right first move whenever the model already has the knowledge or skill needed and just needs the right framing to surface it reliably. It stops being enough when the model genuinely lacks the knowledge or capability required: no amount of clever phrasing teaches a model a fact it was never trained on, or gives it access to information it has no way to see. That's the point where fine-tuning, or supplying external information directly (retrieval, covered later in this handbook), becomes necessary instead.

## How does it work?

A handful of well-established, concrete techniques:

- **Being specific and explicit** about the desired output — format, length, constraints — rather than leaving the model to guess what "good" looks like.
- **Few-shot prompting** — including a small number of example input/output pairs directly in the prompt, letting the model infer the pattern from examples rather than an abstract instruction alone.
- **Chain-of-thought prompting** — asking the model to reason step by step before giving a final answer, which measurably improves performance on tasks that require multi-step reasoning.
- **System or role framing** — setting context about who the model should act as, or what perspective to take, shaping tone and behavior across a whole conversation.

## Example

None of these techniques are free — every one of them adds text to the prompt, and every added token has a real cost, as established in [Tokenization](../nlp/tokenization.md). Comparing three prompts asking for the same underlying task, verified with the same tokenizer used there:

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

vague = "Summarize this."

specific = ("Summarize the following article in exactly 3 bullet points, "
            "each under 15 words, focusing only on financial impact.")

few_shot = specific + """

Example 1:
Article: Company X reported record profits this quarter, up 20% year over year, driven by strong overseas sales.
Summary:
- Profits rose 20% year over year
- Growth driven by overseas sales
- Record quarterly performance reported

Example 2:
Article: Company Y announced layoffs affecting 10% of staff after missing revenue targets for two consecutive quarters.
Summary:
- 10% of staff laid off
- Revenue targets missed twice
- Cost-cutting measure announced
"""

for name, text in [("vague", vague), ("specific", specific), ("few_shot", few_shot)]:
    print(name, len(enc.encode(text)))
```

```text
vague     5
specific  25
few_shot  128
```

The vague prompt costs 5 tokens and leaves the model to guess at length, format, and focus. The specific version costs 5x more tokens but removes that guesswork. The few-shot version costs over 25x the vague prompt's tokens, in exchange for showing the model exactly what the output format should look like, rather than describing it. This is the real trade-off prompt engineering constantly makes: more reliable, more constrained output, in exchange for more tokens — and therefore more cost and more latency — on every single request.

## Where is it used?

Any interaction with an LLM through its text interface: chat applications, coding assistants, content generation pipelines, and as the first line of defense in production LLM systems before reaching for fine-tuning or retrieval.

## Advantages

- **Fast to iterate** — a prompt change takes effect immediately, with no training run required.
- **No additional infrastructure or data collection needed**, unlike fine-tuning.
- **Works with any model behind an API**, including ones you have no ability to retrain at all.

## Limitations

- **Cannot teach a model something it was never trained on** — no phrasing recovers information the model genuinely doesn't have.
- **Every technique that improves reliability adds tokens**, and therefore cost and latency, as shown directly above.
- **Prompts don't transfer cleanly across models.** A prompt tuned for one model version often needs re-tuning on another, since different models respond differently to the same phrasing.

## Production considerations

- **Token cost from prompt engineering techniques compounds at scale.** A few-shot prompt that costs 25x a vague one costs 25x as much on every single request, not just once.
- **Prompts embedded directly in application code are a common source of silent behavior drift** when a model provider updates its underlying model — the same prompt can start behaving differently with no code change at all.
- **Prompt changes deserve the same evaluation discipline as a model change.** A prompt is effectively part of the system's behavior, and an untested prompt change can silently regress quality in production the same way an untested code change can.

## Common mistakes

- **Assuming a vague prompt failing means the model "can't do the task,"** without trying a more specific or few-shot version first.
- **Treating a prompt engineered against one model as portable** to a different model or a new version of the same one, without re-checking it.
- **Reaching for fine-tuning or retrieval before actually trying to improve the prompt**, when the model may have already had the needed capability all along.

## Interview questions

### Basic

- What is prompt engineering, and why can it change a model's output without changing the model itself?
- What is few-shot prompting?

### Intermediate

- Why does adding few-shot examples to a prompt increase cost, and is that trade-off usually worth it?
- When does prompt engineering stop being enough, and what comes next?

### Advanced

- Why might a prompt that works well on one model version behave differently after the provider updates the underlying model?
- How would you evaluate whether a prompt change actually improved a production system's output, rather than just seeming better on a few examples?
