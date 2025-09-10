## Guidelines for LLM collaboration on this LaTeX paper

**Goal:** help write a clear, concise, truthful academic paper (Overleaf-ready LaTeX). Work from `paper_structure.md` and `project_description.txt` and follow precise tasks the user gives.

## 1 — Core rules (non-negotiable)
1. **Only assert what you are *absolutely certain* about.**  
   - If you are uncertain, do **not** write the claim into the paper. Instead add a clear, short uncertainty note (see Comment format).
2. **Do not invent evidence, results, or citations.** If a claim needs support, request the source or leave a TODO comment.
3. **No big structural changes** without explicit instruction.

## 2 — How to write / edit files
- Produce **valid, compilable LaTeX** that runs on Overleaf. respect the project's existing preamble and style.
- Follow the **existing style**. If style is unclear, add one concise comment asking for clarification.
- The user will select the images for the plots and put them in appropriate folders.
sections should be properly titled and numbered.
use labels for referencing.
plots should be EXACTLY where they are. use [H] in figure environment.

## 3 — Commenting conventions (when and how)
Use LaTeX comments (`%`) with prefixes so the user can find them quickly.

- **High-value comments only.** Add comments **occasionally** and only when they:
  - Point out factual gaps, missing references, or logical errors.
  - Propose small, high-leverage improvements (not stylistic nitpicks).
  - Ask a precise question needed to proceed.
- **Comment format (strict):**
  - `%LLM-COMMENT: <one-sentence reason>.`  
  - `%LLM-UNCERTAIN: <short list of what is uncertain / what is needed>.`  
  - `%LLM-TODO: <concrete action the user must take or resource to provide>.`
- Do **not** use comments to repeat or summarize text already present.

## 4 — When uncertain
- If you cannot be certain about a factual statement, do **not** add it to the manuscript. Instead:
  - Add `%LLM-UNCERTAIN:` with 1–3 bullets of exactly what is missing (data, reference, clarification).
  - Optionally add a suggested phrasing that is hedged (e.g., "If X holds, then...") but mark it with `%LLM-UNCERTAIN`.
- If asked to propose text despite limited certainty, clearly label it with `%LLM-UNCERTAIN:` and provide the source types needed to confirm.

## 5 — Citations and bibliography
- Use the **existing bibliography system** lit.bib file
- **Do not fabricate citations.** If you suggest adding a citation, include:
  - The exact bibliographic fields you expect (author, year, title, DOI) and/or ask the user to supply the `.bib` entry.

## 7 — Language & tone
- Audience: **professors in CS or Physics.** Be precise, concise, and formal but clear.
- Write for clarity and truth-seeking. Avoid flourish; prefer direct statements of method, result, and limitation. Limitations in approaches used should be mentioned from the start.
- **Do not produce summaries** for the user — the user will read edits directly. Only add short, actionable comments.

## 8 — Interaction & task handling
- The user will give **precise tasks**. When given a task:
  1. Implement the requested change directly in the target file(s).
  2. If something prevents completion (missing data, ambiguous instruction), add a single `%LLM-UNCERTAIN:` comment and a one-line suggested resolution.
  3. Provide a one-line suggested commit message in a `%LLM-COMMENT:` if helpful.
- If the task is large, perform the work requested and stop; do not ask for confirmation mid-task. If you couldn't complete everything, leave `%LLM-TODO:` items ordered by priority.

## 9 — Error handling and critiques
- If you find logical errors, unsupported claims, or weak evidence in the manuscript:
  - Add **one concise** comment describing the problem and, if possible, a minimal corrective suggestion.
- Do not assert that something is "wrong" unless you can be certain. Prefer "insufficient evidence".
