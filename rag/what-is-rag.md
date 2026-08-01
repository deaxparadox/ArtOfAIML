# What is RAG

## What is it?

RAG (Retrieval-Augmented Generation) is a technique for improving an LLM's output by retrieving relevant external information and inserting it into the prompt, rather than relying solely on what the model learned during training.

Think of it as giving an open-book exam to a model that would otherwise rely purely on memory. The model still does the reasoning and writing, but it's handed the specific relevant pages of the book right before answering, instead of needing to have memorized the whole book in advance.

## Why does it exist?

[Prompt Engineering](../llms/prompt-engineering.md) already established that no phrasing recovers information a model was never trained on. RAG exists to solve exactly that limitation without retraining: instead of trying to get the needed information into the model's weights, RAG fetches it at request time and hands it directly to the model as part of the prompt.

This solves several concrete problems at once: a model has a training cutoff and knows nothing about anything after it; a model has no access to private or internal data — a company's own documents — that was never part of its training data; and even for facts a model does technically know, retraining it every time an answer changes is far more expensive than updating a searchable document store.

**RAG vs. fine-tuning vs. prompting alone, as a real decision:** RAG is the right tool specifically when the need is "give the model access to current or private facts it can look up," not "teach the model a new skill or behavior." Fine-tuning is expensive and best suited to changing how a model behaves — tone, output format, a task-specific skill — rather than what facts it knows at any given moment. Prompting alone, from the previous chapter, works when the model already has the needed knowledge and just needs the right framing. RAG specifically targets the case where the needed knowledge doesn't live in the model's weights at all, and needs to be supplied fresh, per request.

## How does it work?

At a high level — each step gets its own chapter later in this section:

1. A corpus of documents is broken into smaller chunks (Chunking).
2. Each chunk is converted into an embedding (from [Embeddings](../embeddings/embeddings.md)) and stored in a searchable index.
3. At query time, the user's question is also embedded, and the most similar chunks are retrieved (Retrieval), using the similarity metrics already covered in [Similarity](../embeddings/similarity.md).
4. Retrieved chunks are sometimes reranked for relevance before use (Reranking).
5. The retrieved chunks are inserted into the prompt alongside the user's question, and the model generates its answer using both, applying the same techniques from [Prompt Engineering](../llms/prompt-engineering.md).
6. The whole pipeline's output quality needs to be measured (Evaluation).

## Example

The retrieval half of RAG, verified directly — given a tiny knowledge base and a question, find the most relevant document by embedding both and comparing similarity:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "The company's return policy allows returns within 30 days of purchase with a valid receipt.",
    "Our support team is available Monday through Friday, 9am to 6pm Eastern Time.",
    "Shipping typically takes 3 to 5 business days within the continental United States.",
    "The warranty covers manufacturing defects for up to one year from the date of purchase.",
]

query = "How long do I have to return something I bought?"

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])

sims = cosine_similarity(query_vector, doc_vectors)[0]
for doc, score in sorted(zip(documents, sims), key=lambda p: -p[1]):
    print(round(score, 3), "-", doc)
```

```text
0.244 - The company's return policy allows returns within 30 days of purchase with a valid receipt.
0.114 - Shipping typically takes 3 to 5 business days within the continental United States.
0.097 - Our support team is available Monday through Friday, 9am to 6pm Eastern Time.
0.094 - The warranty covers manufacturing defects for up to one year from the date of purchase.
```

The return-policy document scores clearly highest, well ahead of the other three — retrieval correctly found the relevant document from a question phrased quite differently from the document's own wording ("how long do I have to return" vs. "returns within 30 days"). This is the retrieval half of RAG working as intended. The generation half — a model writing an answer using this retrieved document as context — is exactly the prompting mechanism from [Prompt Engineering](../llms/prompt-engineering.md), just with retrieved content standing in for a human-written instruction.

## Where is it used?

Customer support assistants answering from internal documentation, code assistants referencing a private codebase, question-answering systems over a company's own documents, and any application that needs an LLM's answers to reflect current or private information the model was never trained on.

## Advantages

- **Updates instantly** — updating the document store takes effect immediately, unlike retraining a model.
- **Grounds answers in actual source material**, which can also be cited or shown to the user, unlike a model's own memorized knowledge.
- **Gives access to private or internal data** without ever needing to include it in model training at all.

## Limitations

- **Only as good as the retrieval step.** If the wrong document is retrieved, the model is working from the wrong context, no matter how good the model itself is.
- **Adds real latency and cost on top of the model call itself** — an embedding step, a search step, and a longer prompt (more tokens, as covered in [Tokenization](../nlp/tokenization.md)) all happen before generation even starts.
- **Doesn't teach the model a new skill or reasoning pattern** — it only supplies facts. A task that needs the model to reason or behave differently, not just know more, needs fine-tuning or better prompting instead.

## Production considerations

- **Retrieval quality sets a hard ceiling on answer quality.** A system can have an excellent model and still produce bad answers if the retrieved context is irrelevant or incomplete — which makes retrieval, not the model, the first place to debug a bad RAG answer.
- **Keeping the document index in sync with the actual source of truth is an ongoing operational job**, not a one-time setup step. Stale retrieved content produces confidently wrong answers.
- **Every RAG query costs more than a plain prompt** — an embedding call, a search operation, and a longer generation prompt all add latency and cost a simpler system wouldn't have.

## Common mistakes

- **Assuming a wrong or irrelevant answer is a model problem and trying to fix it by changing the prompt**, when the actual retrieved context was wrong to begin with.
- **Treating retrieved context as automatically correct and complete**, when it's only as good as what's in the document store and how well retrieval matched the query to it.
- **Not measuring the retrieval step's accuracy separately from the final answer's quality.** A good final answer can happen despite bad retrieval, or a bad answer despite good retrieval — conflating the two makes debugging much harder.

## Interview questions

### Basic

- What problem does RAG solve that prompting alone can't?
- What are the two main halves of a RAG system?

### Intermediate

- Why is RAG usually cheaper and faster to update than fine-tuning a model on new information?
- What does it mean for a RAG system's answer quality to have a "ceiling" set by retrieval?

### Advanced

- A RAG-based support assistant gives a wrong answer. How would you determine whether the problem is in retrieval or in generation?
- When would fine-tuning be a better fit than RAG, even though RAG is cheaper to update?
