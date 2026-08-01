# Chunking

See [What is RAG](what-is-rag.md) for where this fits in the overall pipeline.

## What is it?

Chunking is the process of splitting a document into smaller pieces before embedding and indexing it for retrieval — deciding where to cut a long document into the units that actually get searched and retrieved.

If RAG hands a model relevant pages of a book, chunking is the decision of how big each "page" is. Too big, and a page buries the one relevant sentence in a lot of irrelevant material; too small, and a page might cut a thought in half, losing the context needed to make sense of it.

## Why does it exist?

[Embeddings](../embeddings/embeddings.md) compress a piece of text into a single fixed-size vector. A whole long document compressed into one vector loses most of its specific detail, since the vector has to represent everything in the document at once, diluting anything specific into an average. And on the retrieval side, [What is RAG](what-is-rag.md) established that only what's actually inserted into the prompt matters — inserting an entire long document to answer one specific question wastes a large number of tokens (real cost, established in [Tokenization](../nlp/tokenization.md)) on mostly irrelevant text. Chunking exists to solve both problems: break a document into pieces small enough that each piece's embedding actually represents its specific content well, and small enough that retrieving a few chunks costs a reasonable number of tokens instead of an entire document.

**Chunk size and overlap, as a real decision:** smaller chunks give more precise retrieval — a chunk's embedding stays tightly focused on one specific piece of information — but risk losing context needed to understand that piece in isolation. Larger chunks preserve more context but dilute the embedding and retrieve more irrelevant material per hit, along with costing more tokens. Overlapping chunks — each chunk sharing a bit of text with its neighbor — is a common compromise, reducing the odds that a critical piece of context gets fully separated at exactly a chunk boundary, at the cost of storing and indexing somewhat more total text.

## How does it work?

**Fixed-size chunking** splits by a set number of tokens (not characters, since tokens are what embeddings and the LLM prompt actually consume), often with overlap between consecutive chunks. **Structure-aware chunking** splits along natural document boundaries — paragraphs, sections, headings — instead of an arbitrary fixed size, so a chunk doesn't cut a coherent thought in half.

## Example

Fixed-size, token-based chunking with overlap, verified directly:

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = (
    "Our return policy allows customers to return most items within 30 days of purchase. "
    "A valid receipt or proof of purchase is required for all returns. "
    "Refunds are issued to the original payment method within 5 to 7 business days. "
    "Items marked as final sale cannot be returned or exchanged under any circumstances."
)

tokens = enc.encode(text)
chunk_size, overlap = 20, 5
chunks, start = [], 0
while start < len(tokens):
    chunks.append(enc.decode(tokens[start:start + chunk_size]))
    start += chunk_size - overlap

for c in chunks:
    print(repr(c))
```

```text
'Our return policy allows customers to return most items within 30 days of purchase. A valid receipt or'
'. A valid receipt or proof of purchase is required for all returns. Refunds are issued to the'
'unds are issued to the original payment method within 5 to 7 business days. Items marked as'
' days. Items marked as final sale cannot be returned or exchanged under any circumstances.'
'.'
```

Two real problems show up directly, unplanned. Chunk 2 starts with `'unds are issued to the'` — the token boundary split the word "Refunds" in half, right where chunk 1 ended with "...Refunds are issued to the" and chunk 2 picked up from "unds" onward. And the final chunk is just a stray period, a tiny, nearly empty leftover fragment from a chunk size that didn't evenly divide the text. Both are exactly the risks fixed-size chunking carries: it cuts on token counts, with no awareness of where a word, sentence, or thought actually ends.

## Where is it used?

Every RAG pipeline's ingestion step, before any embedding or indexing happens — code documentation search, internal knowledge base assistants, and any system retrieving from a document collection too large to hand a model in full.

## Advantages

- **Makes a large document collection searchable at the granularity retrieval actually needs** — a specific relevant passage, not an entire file.
- **Keeps individual chunk embeddings focused**, rather than diluted across an entire long document.
- **Directly controls token cost per retrieved result**, since chunk size determines exactly how many tokens each retrieved piece costs.

## Limitations

- **Fixed-size chunking has no awareness of meaning or structure** — as the example shows directly, it can split a word, a sentence, or a critical piece of context exactly in half.
- **Structure-aware chunking avoids that problem but produces uneven chunk sizes**, since paragraphs and sections aren't uniform lengths — harder to reason about token cost per chunk in advance.
- **Overlap reduces the chance of losing context at a boundary but doesn't eliminate it**, and increases the total amount of text stored and indexed, since overlapping regions get embedded and stored more than once.

## Production considerations

- **Chunk size interacts directly with retrieval cost and prompt cost.** Smaller chunks mean more of them are needed to cover the same content, while larger chunks each cost more tokens once retrieved — there's no universally correct size, only a trade-off tuned to a specific corpus and model's context window.
- **Re-chunking a document collection after changing the chunk size or strategy means re-embedding and re-indexing everything**, not just a config change — a real, sometimes expensive operation on a large existing corpus.
- **A word or fact split across a chunk boundary** (as in this chapter's example) can make that fact effectively unretrievable, even though it's present somewhere in the underlying document — worth checking directly, not just assuming chunking "worked," when a genuinely present fact isn't showing up in retrieval results.

## Common mistakes

- **Choosing a chunk size without checking it against actual example queries and documents**, rather than treating it as a solved default.
- **Assuming overlap eliminates boundary problems entirely**, rather than just reducing their likelihood, as this chapter's example shows they can still occur.
- **Chunking purely by character count instead of token count**, producing chunks with very different actual token costs even though they look similarly sized as text.

## Interview questions

### Basic

- What is chunking, and why is it necessary before embedding a large document collection?
- What's the difference between fixed-size and structure-aware chunking?

### Intermediate

- Why might a smaller chunk size improve retrieval precision but hurt the model's ability to understand the retrieved context?
- Why is overlap used between chunks, and what problem does it only partially solve?

### Advanced

- A fact you know is present in a document isn't being retrieved by your RAG system. How would chunking be involved in diagnosing that?
- Why does changing a chunking strategy require re-processing an entire existing document collection, rather than just changing a setting going forward?
