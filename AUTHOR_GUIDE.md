# AUTHOR_GUIDE.md

# Author Guide

This document defines **how every chapter should be written**.

It is the writing standard for this repository.

---

# Purpose

Every chapter should teach concepts clearly enough that readers understand both **how something works** and **why it works**.

The objective is long-term understanding, not quick memorization.

---

# Target Audience

Assume the reader:

- knows Python
- has basic backend knowledge
- is learning AI Engineering seriously

Do not assume advanced ML knowledge.

---

# Writing Philosophy

Write like an experienced engineer teaching another engineer.

Focus on:

- clarity
- reasoning
- intuition
- practical engineering

Avoid unnecessary complexity.

The goal is not just to explain what something is and how it works. Every chapter should also help the reader answer: **how would an experienced engineer think about this?** That means surfacing why it was built, what it's traded off against, and what actually goes wrong with it in production — not just the mechanism.

Many topics naturally involve choosing between competing approaches. When a topic genuinely involves a decision like that, explain how an experienced engineer would think it through: when to use it, when to avoid it, what trade-off is being accepted, and what would change the decision. The goal is to build the reader's judgment, not to hand them a recommendation. If a topic doesn't involve a real decision, don't invent one just to cover this point.

Write conversationally, as one engineer explaining a topic to another — not like reference documentation. The reader should feel guided, not lectured.

---

# Language Rules

- Write in simple technical English.
- Prefer short, clear sentences.
- Explain technical terms before using them.
- Avoid marketing language.
- Avoid AI-generated sounding phrases.
- Avoid unnecessary buzzwords.
- Use active voice.
- Avoid unnecessary adjectives, motivational language, repetitive sentence structures, filler paragraphs, exaggerated claims, and generic summaries that don't add new information.

---

# Explanation Rules

Every chapter follows this section order:

1. **What is it?** — a precise definition.
2. **Why does it exist?** — the problem it was created to solve. Cover what existed before, why that wasn't enough, and why engineers still choose this approach today.
3. **How does it work?** — the mechanism. Include a mental-model analogy when the concept is abstract (e.g. "think of a model as a learned function, not a hand-written one").
4. **Example** — a small, focused example illustrating the mechanism just described. Its own heading, separate from "How does it work?".
5. **Where is it used?** — real domains and use cases.
6. **Advantages** — genuine strengths, and why an experienced engineer would reach for this over the alternatives.
7. **Limitations** — genuine weaknesses, and when an alternative is the better choice. Nothing is presented as universally good — state the trade-off plainly.
8. **Production considerations** — the failure modes engineers actually hit (scaling, latency, drift, monitoring gaps, debugging difficulty, operational cost, maintenance burden), not a generic best-practices checklist.
9. **Common mistakes** — mistakes engineers actually make, ideally as a concrete observation ("beginners often assume X; in practice Y matters more") rather than a generic warning.
10. **Interview questions** (when applicable) — grouped as Basic / Intermediate / Advanced.

Include a genuine engineering-intuition observation wherever one exists for the topic — a real trade-off, a common misconception, a production lesson. Don't manufacture one for the sake of having one; a section with no natural insight to add is fine left as-is. A forced observation reads as filler, which is exactly what this guide asks you to avoid.

---

# Code Examples

Use practical examples.

Examples should:

- compile
- be minimal
- explain one concept
- avoid unnecessary complexity

---

# Diagrams

Default to a Mermaid diagram whenever a workflow, architecture, pipeline, lifecycle, or interaction is being explained — not only when convenient.

Prefer a diagram over several paragraphs whenever one exists that would replace them.

**Rendered plots** (e.g. Matplotlib, Plotly output) are a separate case from diagrams: save the image to an `assets/` subfolder alongside the chapter, named descriptively (`assets/matplotlib-bias-variance-fit.png`), and embed it with a real alt-text description of what it shows.

---

# GitHub Markdown

Always generate valid GitHub Markdown.

Use:

- headings
- tables
- callouts
- code blocks
- Mermaid diagrams
- relative links

Avoid raw HTML unless necessary.

---

# Internal Links

Whenever another topic already exists:

Reference it using relative Markdown links instead of repeating the explanation.

---

# Review Checklist

Before finishing a chapter verify:

- Technical correctness
- Clear explanations
- Consistent terminology
- Proper Markdown formatting
- Working internal links
- No duplicate explanations
- No AI filler
- Trade-offs are stated, not just advantages
- Production considerations describe real failure modes, not a generic checklist
- Interview questions (when applicable) are grouped Basic / Intermediate / Advanced
- Any engineering-intuition observation present is genuine, not manufactured to fill a quota
- Where a topic involves a real decision between approaches, the chapter explains how an experienced engineer would reason through it — without inventing a decision where none exists