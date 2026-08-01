# LLM Evaluation / Benchmarks

See [Evaluation](../rag/evaluation.md) for the retrieval-vs-generation split this chapter focuses in on the generation half of.

## What is it?

LLM evaluation is the practice of measuring how well a language model performs — on general capability, via standardized benchmarks (MMLU, HellaSwag), on a specific task, via a curated test set, or on subjective qualities like helpfulness, via human evaluation or LLM-as-judge scoring, already introduced in [Evaluation](../rag/evaluation.md).

[Evaluation](../rag/evaluation.md) separated retrieval quality from generation quality specifically because a single end-to-end score doesn't say which half to fix. This chapter is the generation half of that story, applied more broadly: a single "the model seems good" impression isn't enough — specific, measurable answers to specific questions (does it reason well, does it follow instructions, is it accurate on this exact task) are what evaluation is actually for.

## Why does it exist?

[Prompt Engineering](prompt-engineering.md) and [Fine-Tuning](fine-tuning.md) both describe ways to change a model's behavior — but changing something requires being able to measure whether the change actually helped, the same evaluation-as-feedback-loop principle already established in [ML Workflow](../machine-learning/ml-workflow.md). LLM evaluation exists to give that principle tools specific to language models: an LLM's output is open-ended text, not a single number or class label the way a classical model's output usually is, so it needs evaluation approaches beyond a simple accuracy score.

**Benchmark-based vs. task-specific vs. LLM-as-judge evaluation — a real choice depending on the actual question.** Standardized benchmarks are useful for comparing general capability across models, but a model can score well on a broad public benchmark and still perform poorly on a narrow, specific use case that benchmark never represented. Task-specific evaluation — a test set built for the exact deployment use case — is more relevant, but requires building that test set yourself. LLM-as-judge is a practical middle ground for scoring open-ended output at scale where a single "correct answer" doesn't exist, but it inherits the judge model's own blind spots, as [Evaluation](../rag/evaluation.md) already noted.

## How does it work?

**Standardized benchmarks** are a fixed, public set of test questions covering a general capability, with a known scoring method — useful for comparing model versions or providers at a glance. A real, documented risk: a model can be specifically optimized to score well on public benchmarks without that improvement generalizing to real, novel use cases. **Task-specific evaluation** curates examples representative of the actual deployment use case, scored against a reference answer. **LLM-as-judge** has another model score a candidate response against stated criteria — helpfulness, correctness, tone.

## Example

Comparing two automatic ways of scoring a candidate answer against a reference — exact match, and the semantic similarity approach from [Embeddings](../embeddings/embeddings.md) — on answers written to illustrate the method, not claimed to be real model output:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

reference = "The capital of France is Paris."
candidates = [
    "The capital of France is Paris.",       # exact match
    "Paris is the capital city of France.",  # correct, reworded
    "France's capital city is Paris.",       # correct, reworded
    "The capital of France is Lyon.",        # wrong
]

for c in candidates:
    print(c.strip().lower() == reference.strip().lower())

tfidf = TfidfVectorizer()
X = tfidf.fit_transform([reference] + candidates)
svd = TruncatedSVD(n_components=2, random_state=0)
embeddings = svd.fit_transform(X)
sims = cosine_similarity(embeddings)[0][1:]
for c, s in zip(candidates, sims):
    print(c, round(s, 3))
```

```text
exact match: True, False, False, False
semantic similarity: 1.000, 0.924, 0.715, 0.875
```

Exact match penalizes both correct paraphrases identically to the wrong answer — all three score `False`, a clear failure to distinguish "worded differently but right" from "wrong." Semantic similarity does better on one paraphrase (0.924) but reveals its own serious blind spot: the *wrong* answer ("Lyon") scores 0.875 — higher than the *correct* paraphrase "France's capital city is Paris" at 0.715 — because it shares more surface words with the reference ("the capital of France is X"), even though it's factually incorrect. Neither automatic metric reliably measures correctness here. That's precisely the gap LLM-as-judge and human evaluation exist to fill: an evaluator that understands what the answer actually claims, not just how many words it shares with a reference.

## Where is it used?

Comparing candidate models or prompt/fine-tuning changes before deployment, tracking whether a production LLM system's output quality holds steady over time, and reporting model capability to stakeholders in specific, checkable terms rather than a vague impression.

## Advantages

- **Turns "does this seem better" into a specific, comparable measurement**, the same benefit [Evaluation](../rag/evaluation.md) already established for RAG pipelines generally.
- **Standardized benchmarks enable quick, broad comparison** across models or providers without building custom test infrastructure.
- **LLM-as-judge scales evaluation of open-ended output** to volumes full human review can't reach.

## Limitations

- **Automatic metrics can be actively misleading, not just imprecise** — as the example shows directly, a factually wrong answer scored higher than a correct one under a similarity-based metric.
- **Standardized benchmarks can be gamed or contaminated**, especially once their exact content becomes a known target for training, weakening what a high score actually signals about real-world generalization.
- **LLM-as-judge inherits its judge model's own limitations and biases** — a scalable proxy for human judgment, not a neutral ground truth.

## Production considerations

- **A production system needs task-specific evaluation, not just a general benchmark score**, since a strong benchmark result says little about the exact use case the system will actually face.
- **Automatic metrics need periodic sanity-checking against human judgment**, precisely because they can fail silently — a metric that looks stable might be consistently rewarding the wrong things, undetected.
- **Evaluation needs to run on every meaningful change** — a new prompt, a new model version, a fine-tuning update — the same discipline already established in [ML Workflow](../machine-learning/ml-workflow.md) and [Evaluation](../rag/evaluation.md).

## Common mistakes

- **Trusting a similarity-based automatic metric as a stand-in for correctness**, without checking, as this chapter's example shows, that it can rank a wrong answer above a right one.
- **Reporting only a general benchmark score** as evidence a model will perform well on a specific, different downstream task.
- **Treating an LLM judge's score as objective ground truth**, rather than a useful but imperfect, biased proxy for human judgment.

## Interview questions

### Basic

- Why is a simple accuracy score not sufficient for evaluating an LLM's open-ended output?
- What is LLM-as-judge evaluation?

### Intermediate

- Why did the semantic similarity metric in this chapter's example score a wrong answer higher than a correct one?
- Why can a strong score on a general benchmark fail to predict performance on a specific production use case?

### Advanced

- Design an evaluation strategy for a production LLM feature that needs both broad capability comparison and task-specific correctness checking. What would each part actually measure?
- What does "benchmark contamination" mean, and how would you detect that a reported benchmark score is misleading?
