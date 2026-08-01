# Tokenization

## What is it?

Tokenization is the process of splitting raw text into smaller units — **tokens** — that a model can work with numerically. A token can be a whole word, a piece of a word (a subword), or a single character, depending on the scheme used.

Think of raw text as a single, unbroken strip of paper. Tokenization is cutting that strip into pieces small enough for a model to hold one at a time — the real question is exactly where to make each cut.

## Why does it exist?

Models operate on numbers, not raw text — the same gap [Feature Engineering](../machine-learning/feature-engineering.md) covered for categorical data, just at the scale of an entire language's vocabulary instead of a handful of categories. Before any model can process a sentence, it needs to be broken into discrete units that each map to an integer ID, and eventually an embedding vector.

Early tokenizers just split on whitespace, treating each word as a unit. This breaks down quickly: punctuation gets stuck to words ("pieces." isn't conceptually "pieces" plus a period), long or rare words balloon a model's vocabulary, and a word never seen during training has no entry in the vocabulary at all — a hard failure, with no graceful way to represent it. Modern subword tokenization schemes (Byte-Pair Encoding, used below) exist specifically to solve this: common words and word-pieces get merged into single tokens, while anything rare or entirely novel still gets represented, by falling back to smaller and smaller known pieces.

**Word-level vs. subword vs. character-level, as a real decision:** word-level tokenization is simple and interpretable but has a hard vocabulary-size ceiling and cannot represent an unseen word at all. Character-level tokenization has a tiny, fixed vocabulary and can represent any input, but produces very long sequences, since a single character rarely carries meaning on its own. Subword tokenization — the modern default, used by virtually every current LLM — is the practical middle ground: common words and patterns stay as single tokens (keeping sequences short), while anything rarer still gets represented by falling back to smaller pieces instead of failing outright.

## How does it work?

A tokenizer is trained once, ahead of time, on a large text corpus, building a fixed vocabulary of the most useful subword pieces. Byte-Pair Encoding (BPE) does this by repeatedly merging the most frequent adjacent pair of symbols into a new token, starting from individual bytes and building upward. At encoding time, that fixed vocabulary is used to split new text into the largest matching known pieces, and each resulting piece maps to an integer ID — the actual numeric representation a model consumes.

## Example

Verified with `tiktoken`, the BPE tokenizer used by OpenAI's models:

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "Tokenization splits unpredictability into pieces."
tokens = enc.encode(text)
print(len(text.split()), len(tokens))  # -> 5 words, 8 tokens

for t in tokens:
    print(t, repr(enc.decode([t])))
```

```text
3404  'Token'
2065  'ization'
41567 ' splits'
44696 ' unpredict'
2968  'ability'
1139  ' into'
9863  ' pieces'
13    '.'
```

The two longer, rarer words — "Tokenization" and "unpredictability" — each split into two subword pieces. The three short, common words, and the period, each stay as a single token. That's the trade-off from the previous section, playing out directly: common patterns collapse into single tokens; anything rarer costs more tokens, but remains representable.

That last point matters more than it looks — an entirely made-up word still encodes cleanly, with no error and no "unknown" placeholder:

```python
nonsense = "zxqvlorpTHE"
tokens = enc.encode(nonsense)
print(enc.decode(tokens) == nonsense)  # -> True
```

Decoded, the tokens (`'zx'`, `'qv'`, `'lor'`, `'p'`, `'THE'`) reconstruct the input exactly — falling back to smaller and smaller pieces rather than failing, exactly the property that made subword tokenization the practical default.

## Where is it used?

Every text-based model, from a basic bag-of-words classifier to the LLMs behind modern chatbots. It's also the actual unit LLM API pricing and context-window limits are measured in — "this model's context window is 128k tokens" is a direct, practical consequence of the scheme covered in this chapter, not an abstract implementation detail.

## Advantages

- **Handles any input text**, including words never seen during training, by falling back to smaller known pieces instead of failing.
- **Keeps sequences reasonably short** by representing common words and patterns as single tokens.
- **A fixed, learned vocabulary** — the same tokenizer produces the same token IDs for the same text every time, which is what makes results reproducible.

## Limitations

- **Token boundaries don't always align with what a person considers a meaningful unit.** A word can be split in a way that looks arbitrary, purely because of how frequently that substring appeared in the tokenizer's training corpus.
- **Token count, not word or character count, determines cost and context-window usage** for most LLM APIs — the same sentence can cost meaningfully more or fewer tokens depending on phrasing, something that isn't obvious without actually running the tokenizer.
- **Different models use different tokenizers with different vocabularies** — tiktoken alone ships several named encodings. Token counts, and even the exact tokens produced, aren't portable from one model's tokenizer to another's.

## Production considerations

- **Token counts, not word counts, are the real unit of cost and capacity** for LLM-based systems — estimating usage or context-window budget from word counts alone is reliably wrong, sometimes significantly so, as the example above shows directly (5 words, 8 tokens).
- **Some tokenizer libraries fetch their vocabulary and merge-rule files over the network on first use**, caching them locally afterward — worth knowing before deploying into an environment without outbound internet access, where that first call would otherwise fail unexpectedly.
- **Swapping the underlying model in a pipeline can silently swap its tokenizer too**, changing token counts, costs, and context-window behavior for text that hasn't changed at all.

## Common mistakes

- **Estimating LLM cost or context usage from word or character counts** instead of actually running the tokenizer — as shown above, a five-word sentence can cost eight tokens, not five.
- **Assuming a word that "looks simple" tokenizes to one token.** Length and familiarity to a human aren't what determines a token boundary — corpus frequency during the tokenizer's training is.
- **Mixing tokenizers across models without re-checking assumptions.** A token budget or prompt-engineering trick tuned against one model's tokenizer doesn't necessarily transfer to a model using a different one.

## Interview questions

### Basic

- What is a token, and why do models need text converted into tokens?
- Why doesn't naive whitespace splitting work well as a tokenization scheme?

### Intermediate

- Why can two sentences with the same number of words have very different token counts?
- How does subword tokenization handle a word it's never seen before, without failing?

### Advanced

- Why does token count, not word count, matter for LLM API cost and context-window limits?
- What would you expect to happen to token counts, cost, and context-window behavior if a pipeline swapped its underlying model but not its tokenizer, or vice versa?
