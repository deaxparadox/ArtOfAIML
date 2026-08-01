# Text Cleaning

See [Tokenization](tokenization.md) for the step this chapter comes before.

## What is it?

Text cleaning (or preprocessing) is the step of normalizing raw text before tokenization and modeling — removing or standardizing the parts of text that carry noise rather than signal: inconsistent casing, HTML markup, stray punctuation, extra whitespace, and similar artifacts.

Raw text scraped from the real world is like a photo taken in bad lighting with smudges on the lens. Text cleaning is wiping the lens and correcting the exposure before trying to recognize what's in the picture — it doesn't change the subject, it just removes what makes it harder to see clearly.

## Why does it exist?

[Feature Engineering](../machine-learning/feature-engineering.md) already established that a model can only learn from information it receives in a usable form. Text cleaning is the text-specific version of that principle: without it, meaningless variation — "Apple" vs. "apple" vs. "APPLE," an HTML tag stuck to a word, extra whitespace — gets treated by a naive tokenizer as a completely different, unrelated piece of vocabulary from the "clean" version of the same content, diluting the signal a model could otherwise learn from.

This mattered even more before subword tokenization (covered in [Tokenization](tokenization.md)) became standard. A classical word-level model has no way to learn on its own that "running" and "run" are related — cleaning does that work by hand, ahead of time. Subword tokenizers reduce this problem but don't eliminate it: differently-cased or noisy text still produces different token sequences even under BPE.

**How much cleaning to do is itself a decision, and it depends on the model.** Classical bag-of-words or TF-IDF style models benefit heavily from aggressive cleaning — lowercasing, stopword removal, stemming — because they have no other way to recognize related words as related. Modern subword-tokenized deep models (LLMs) need much lighter cleaning: they can often learn to handle casing and minor noise directly from data at scale, and aggressive cleaning can actively destroy real signal a modern model could have used — "US" the country and "us" the pronoun are not the same word, and lowercasing erases that distinction for good.

## How does it work?

Common operations, roughly from least to most aggressive: stripping HTML tags and URLs; normalizing whitespace; removing punctuation; lowercasing; removing stopwords (very common words like "the," "is," "and" that carry little topic-specific signal for classical models); and reducing words to a base form, either crudely (**stemming**, suffix-stripping like "running" → "run") or more precisely (**lemmatization**, a dictionary-based reduction like "better" → "good").

## Example

Cleaning a small set of product reviews and measuring the vocabulary size before and after — the real effect cleaning has on what a classical model would actually see.

```python
import re

reviews = [
    "Great product!! Really <b>loved</b> it.",
    "great PRODUCT, would buy again",
    "Terrible...  product broke in a week",
]

naive_vocab = set()
for r in reviews:
    naive_vocab.update(r.split())
print(len(naive_vocab))  # -> 16

def clean(text):
    text = re.sub(r"<[^>]+>", "", text)       # strip HTML tags
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)    # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text

cleaned_vocab = set()
for r in reviews:
    cleaned_vocab.update(clean(r).split())
print(len(cleaned_vocab))  # -> 13
```

Before cleaning, `'Great'`, `'great'`, `'PRODUCT,'`, `'product'`, and `'product!!'` are all counted as different, unrelated words — and `'<b>loved</b>'` sits in the vocabulary as one garbled token instead of the real word `'loved'`. After cleaning, all three product mentions collapse into a single `'product'` entry, and `'loved'` recovers cleanly. The vocabulary shrinks from 16 to 13 words across just three short sentences — at real scale, this compounds into a substantial reduction in how much unrelated noise a classical model has to separately learn.

## Where is it used?

Any classical NLP pipeline (bag-of-words, TF-IDF, older sentiment or topic models), and the data-cleaning side of the EDA step from [ML Workflow](../machine-learning/ml-workflow.md) whenever the raw data is text — scraped reviews, support tickets, log messages, social media posts.

## Advantages

- **Directly reduces vocabulary size and noise** for models that can't learn word relationships on their own.
- **Makes downstream analysis more meaningful** — word frequency counts and simple keyword search both improve once meaningless variants of the same word collapse together.
- **Cheap to apply.** Most operations are simple, fast string transformations with no model or training step required.

## Limitations

- **Cleaning can destroy real signal, not just noise.** Lowercasing erases the "US" vs. "us" distinction, and stemming can conflate words that meant different things ("universal" and "university" both crudely stem toward "univers" in some stemmers).
- **What counts as "noise" is model- and task-dependent.** Stopwords are noise for topic classification but can matter for sentiment or grammar-sensitive tasks — there's no universal cleaning recipe that's correct for every downstream use.
- **Modern subword-tokenized models need far less of this** than the classical example above implies. Applying a classical-NLP-era cleaning pipeline to an LLM-based system by default can quietly remove information the model would otherwise have used.

## Production considerations

- **Cleaning logic has to be applied identically at training and inference time** — the same train-serving skew concern already raised in [Feature Engineering](../machine-learning/feature-engineering.md), just for text transformations instead of a fitted scaler.
- **A cleaning step tuned for one kind of text doesn't transfer automatically.** Something built for English product reviews can silently misbehave on a different language's punctuation and casing conventions, or on code and log messages that look nothing like prose.
- **Aggressive, blanket cleaning applied without checking the model's actual sensitivity to it can quietly remove signal that never gets noticed missing** — it doesn't fail loudly, it just leaves the model with slightly less to work with.

## Common mistakes

- **Applying a classical NLP cleaning pipeline in front of a modern LLM-based system by default**, without checking whether that model actually benefits from it or is instead losing usable signal.
- **Cleaning text differently in a training pipeline than in the live serving path**, producing a subtle mismatch between what the model learned on and what it actually sees in production.
- **Treating "clean" as a single, universal target rather than a task-specific decision.** Stopword removal that helps a topic classifier can actively hurt a task where those exact words carry the meaning — negation words like "not" matter a great deal for sentiment analysis.

## Interview questions

### Basic

- What is text cleaning, and why does it matter before tokenization or modeling?
- What's the difference between stemming and lemmatization?

### Intermediate

- Why might lowercasing hurt a model's performance instead of helping it?
- Why does a classical bag-of-words model need more aggressive text cleaning than a modern LLM-based pipeline?

### Advanced

- A production text-classification pipeline's accuracy dropped right after a redeploy, and the only change was to the text-cleaning code. How would you investigate whether cleaning is the cause?
- Why is there no single, universally correct text-cleaning recipe across different NLP tasks?
