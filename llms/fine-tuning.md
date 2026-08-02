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
print(accuracy_score(y_test, scratch_model.predict(X_test)))  # -> 0.825

pretrained_model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=0, warm_start=True)
pretrained_model.fit(X_general, y_general)
print(accuracy_score(y_test, pretrained_model.predict(X_test)))  # -> 0.975, before any task-specific training at all

pretrained_model.set_params(max_iter=100)
pretrained_model.fit(X_small, y_small)
print(accuracy_score(y_test, pretrained_model.predict(X_test)))  # -> 0.925
```

Training from scratch on just 12 examples reaches 82.5% accuracy. The pretrained model — evaluated before it has seen a single task-specific example — already reaches 97.5%, purely by carrying over general structure learned from the larger dataset. Continuing to train it on those same 12 examples then makes it *worse*, down to 92.5% — an honest, if initially surprising, result worth explaining rather than hiding: with only 12 examples to fine-tune on, the model overfits to that tiny set's specific noise, actively damaging the better-generalizing representation the pretrained weights already had. Fine-tuning isn't automatically an improvement layered on top of a pretrained starting point — with too little task-specific data, it can genuinely make a model worse than simply using it zero-shot. This is a real, well-known risk in practice, not an artifact of this toy example: the fix is more fine-tuning data, a smaller learning rate or fewer fine-tuning steps, or in some cases deciding the pretrained model's zero-shot performance was already good enough.

## Where is it used?

Specializing a general-purpose LLM for a consistent brand voice, a domain-specific reasoning style (legal, medical, code), or a structured output format that needs to be reliable at a scale prompting alone can't guarantee.

## Advantages

- **Can teach genuinely new behavior or skill**, not just remind the model of something it already knows how to do.
- **Leverages pretrained general knowledge**, as the example shows directly — a pretrained starting point dramatically outperforms training on the same small amount of data from scratch.
- **Parameter-efficient methods make this affordable** at far less compute and storage cost than updating every weight in a large model.

## Limitations

- **Fine-tuning's benefit isn't guaranteed, and can be negative**, as this chapter's own example shows directly — with too little task-specific data, continued training actively made a better-generalizing pretrained model worse, not just failed to improve it.
- **Requires a labeled, task-specific dataset**, with all the same collection and quality costs already covered in [Feature Engineering](../machine-learning/feature-engineering.md) and [ML Workflow](../machine-learning/ml-workflow.md) — and, as the example shows, a dataset too small to fine-tune on safely is itself a real risk, not just a smaller benefit.
- **A poorly aligned pretrained starting point can actively hurt in a different way** — if the source domain conflicts with the target task rather than complementing it, fine-tuning may not be enough to correct course even with adequate data, independent of the small-data overfitting risk shown here.

## Production considerations

- **A fine-tuned model is now a custom artifact that needs its own versioning, evaluation, and maintenance** — the same reproducibility discipline from [ML Workflow](../machine-learning/ml-workflow.md), now applied to a model that isn't just a hosted API call anymore.
- **Fine-tuning cost and infrastructure needs are real and ongoing** — a fine-tuned model needs retraining as the target task or data shifts, the same way any production model does.
- **The decision to fine-tune should follow from having actually tried prompting and RAG first**, per [What is RAG](../rag/what-is-rag.md)'s own framing, not from assuming fine-tuning is the more "serious" solution by default.

## Common mistakes

- **Reaching for fine-tuning before trying prompt engineering or RAG**, taking on real cost and maintenance burden for a problem a cheaper lever might have solved.
- **Assuming fine-tuning will always improve on a pretrained model's zero-shot baseline**, when this chapter's example shows it can measurably make things worse if the fine-tuning set is too small to safely train on.
- **Fine-tuning on too little data without checking the zero-shot baseline first** — as the example shows, that baseline can already be the better model, and skipping the comparison means never noticing fine-tuning made things worse.

## Interview questions

### Basic

- What's the difference between pretraining and fine-tuning?
- Why does fine-tuning use a much smaller learning rate than pretraining?

### Intermediate

- Why might a pretrained model, with zero task-specific fine-tuning, already outperform a model trained from scratch on a small task-specific dataset?
- What is parameter-efficient fine-tuning, and why does it reduce cost compared to full fine-tuning?

### Advanced

- In this chapter's example, fine-tuning on 12 examples made the model *worse* than its own zero-shot performance. What does that suggest about the relationship between fine-tuning set size and overfitting risk, and how would you mitigate it?
- How would you decide whether a use case needs fine-tuning, versus better prompting, versus RAG — and how would you make sure a fine-tuning decision doesn't get made without first checking the pretrained model's zero-shot baseline?
