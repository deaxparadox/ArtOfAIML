# Fine-Tuning

See [Prompt Engineering](prompt-engineering.md) and [What is RAG](../rag/what-is-rag.md) for the two alternatives this chapter completes the picture alongside.

## What is it?

Fine-tuning is the process of taking a pretrained model and continuing its training on a smaller, task-specific dataset, adjusting its weights so its behavior specializes to a particular task or domain — rather than relying entirely on prompting a general-purpose model as-is.

Continuing [Self-Supervised Learning](../machine-learning/self-supervised-learning.md)'s own "pretraining then fine-tuning" pattern: if pretraining teaches a model general-purpose language understanding from a huge, unlabeled corpus, fine-tuning is the much smaller, targeted follow-up — taking that broadly-capable model and giving it focused additional practice on exactly the kind of task or domain it'll actually be used for.

## Why does it exist?

[Prompt Engineering](prompt-engineering.md) established that prompting can't teach a model something it was never trained on — no phrasing recovers a skill the model genuinely lacks. Fine-tuning exists to be the tool that can: unlike prompting, which works within the model's existing weights, or RAG, which supplies facts the model can look up, fine-tuning actually changes the model's weights, letting it learn a new behavior, tone, or specialized skill directly.

**Fine-tuning vs. prompting vs. RAG — the three-way decision [What is RAG](../rag/what-is-rag.md) already set up.** That chapter established "RAG for facts, fine-tuning for behavior or skill." Fine-tuning is worth its real cost — collecting a labeled dataset, running a training job, and then maintaining a custom model going forward — specifically when the need is a change in *how* the model behaves: a consistent tone, a specialized output format, a domain-specific reasoning pattern that no amount of prompting reliably achieves, and that isn't just a missing fact retrieval could supply. If the need is simply "make the model produce a certain format more reliably," it's worth trying the specificity and few-shot techniques from [Prompt Engineering](prompt-engineering.md) first, before taking on fine-tuning's much higher cost.

## How does it work?

Fine-tuning starts from a pretrained model's weights, not training from scratch, which would forgo the entire benefit of pretraining's massive scale. Training continues on a smaller, labeled, task-specific dataset — typically with a much smaller learning rate than pretraining used, so the model adapts to the new task without forgetting its general capabilities. **Full fine-tuning** updates all of a model's weights, which is expensive even though the dataset is small. **Parameter-efficient fine-tuning** (e.g. LoRA) trains only small additional adapter weights while keeping the pretrained weights frozen, achieving much of the same task-specific improvement at a small fraction of the compute and memory cost.

## Example

A full LLM fine-tuning run isn't feasible to actually execute here, but the core mechanism — start from a model already trained on a general task, then adapt it — is real and verifiable with a small neural network. A model pretrained on 700 general examples, compared against training a fresh model on only 12 task-specific examples:

```python
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_general, y_general = make_moons(n_samples=700, noise=0.15, random_state=0)
X_spec_all, y_spec_all = make_moons(n_samples=400, noise=0.15, random_state=1)
X_test, X_pool, y_test, y_pool = train_test_split(X_spec_all, y_spec_all, test_size=0.9, random_state=0)
X_small, y_small = X_pool[:12], y_pool[:12]  # only 12 task-specific examples

scratch_model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=0)
scratch_model.fit(X_small, y_small)
print(accuracy_score(y_test, scratch_model.predict(X_test)))  # -> 0.700

pretrained_model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=0, warm_start=True)
pretrained_model.fit(X_general, y_general)
print(accuracy_score(y_test, pretrained_model.predict(X_test)))  # -> 0.833, before any task-specific training at all

pretrained_model.set_params(max_iter=100)
pretrained_model.fit(X_small, y_small)
print(accuracy_score(y_test, pretrained_model.predict(X_test)))  # -> 0.833
```

Training from scratch on just 12 examples reaches 70% accuracy. The pretrained model — evaluated before it has seen a single task-specific example — already reaches 83.3%, purely by carrying over general structure learned from the larger dataset. Continuing to train it on the same 12 examples didn't improve on that further here, an honest result worth explaining rather than hiding: in this case, the general pretraining distribution already overlapped enough with the specialized task that most of the benefit came from starting with pretrained weights at all, not from the small amount of additional fine-tuning. How much fine-tuning adds *beyond* a pretrained model's starting point depends entirely on how much the target task actually differs from what the model already learned.

## Where is it used?

Specializing a general-purpose LLM for a consistent brand voice, a domain-specific reasoning style (legal, medical, code), or a structured output format that needs to be reliable at a scale prompting alone can't guarantee.

## Advantages

- **Can teach genuinely new behavior or skill**, not just remind the model of something it already knows how to do.
- **Leverages pretrained general knowledge**, as the example shows directly — a pretrained starting point dramatically outperforms training on the same small amount of data from scratch.
- **Parameter-efficient methods make this affordable** at far less compute and storage cost than updating every weight in a large model.

## Limitations

- **Fine-tuning's marginal benefit isn't guaranteed**, as this chapter's own example shows — when the pretrained model already generalizes well to the target task, additional fine-tuning may add little on top of it.
- **Requires a labeled, task-specific dataset**, with all the same collection and quality costs already covered in [Feature Engineering](../machine-learning/feature-engineering.md) and [ML Workflow](../machine-learning/ml-workflow.md).
- **A poorly aligned pretrained starting point can actively hurt.** If the source domain conflicts with the target task rather than complementing it, a few fine-tuning steps may not be enough to correct course, and can perform worse than training fresh on the same small dataset.

## Production considerations

- **A fine-tuned model is now a custom artifact that needs its own versioning, evaluation, and maintenance** — the same reproducibility discipline from [ML Workflow](../machine-learning/ml-workflow.md), now applied to a model that isn't just a hosted API call anymore.
- **Fine-tuning cost and infrastructure needs are real and ongoing** — a fine-tuned model needs retraining as the target task or data shifts, the same way any production model does.
- **The decision to fine-tune should follow from having actually tried prompting and RAG first**, per [What is RAG](../rag/what-is-rag.md)'s own framing, not from assuming fine-tuning is the more "serious" solution by default.

## Common mistakes

- **Reaching for fine-tuning before trying prompt engineering or RAG**, taking on real cost and maintenance burden for a problem a cheaper lever might have solved.
- **Assuming fine-tuning will always improve on a pretrained model's baseline**, when the actual improvement depends on how much the target task genuinely differs from the pretraining distribution.
- **Fine-tuning on too little data without checking whether the pretrained starting point is even well-aligned with the target task first.**

## Interview questions

### Basic

- What's the difference between pretraining and fine-tuning?
- Why does fine-tuning use a much smaller learning rate than pretraining?

### Intermediate

- Why might a pretrained model, with zero task-specific fine-tuning, already outperform a model trained from scratch on a small task-specific dataset?
- What is parameter-efficient fine-tuning, and why does it reduce cost compared to full fine-tuning?

### Advanced

- In this chapter's example, fine-tuning added no measurable benefit over the pretrained model's zero-shot performance. What would that suggest about the relationship between the pretraining and target distributions, and when would you expect fine-tuning to add more?
- How would you decide whether a use case needs fine-tuning, versus better prompting, versus RAG?
