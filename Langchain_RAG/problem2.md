# Problem Statement — RAG-Powered “Collaboration & Mentor Matcher”

## Background (Context for Professors)
You are given a small “student project database” knowledge base that contains:
- Student project **reports (PDF)**, **summaries (TXT)**, and **metadata (JSON)**
- A few global documents like **Projects**, **Mentors**, and **Criteria** PDFs

You have already completed (or been shown) three notebooks:
1) **Simple RAG** (expected to fail on complex questions)
2) **Multi-Query RAG** (improves recall for multi-intent questions)
3) **Query Translation + Decomposition** (handles multilingual + multi-constraint queries)

Now you will build a small but realistic RAG capability that is *easy to start* but *challenging to do correctly*.

---

## Your Task
Build a RAG pipeline that answers this user request **grounded only in the provided dataset**:

> **“Recommend a 2–3 student collaboration team (with roles) and one suitable mentor for a campus prototype, based on multiple constraints.”**

You must support:
1. **Multi-constraint reasoning**
2. **Multi-intent retrieval** (multiple sub-queries)
3. **At least one multilingual query** (Malay/Arabic/Hindi — your choice)

---

## Core Prompt (The User’s Question)
Your system must answer the following **exact** question:

### Q1 (English — Multi-Constraint)
> We want to build a **RAG-based academic assistant** for a university within **8 weeks**.  
> Recommend **2–3 students** who can collaborate.  
> Constraints:
> - At least **1 student** must have strong **NLP/LLM/RAG** exposure.
> - At least **1 student** must have strong **backend/API integration** exposure (FastAPI/Node/Django/etc.).
> - Prefer students whose work suggests **evaluation/metrics** or **data handling** discipline.
> Also recommend **1 mentor** who is the best fit.
>  
> Provide:
> - Team members + **assigned roles**
> - Mentor recommendation
> - **Evidence**: 2–4 short snippets (bullet points) from the retrieved context for each recommendation
> - If evidence is insufficient: say **“Not found in the provided data.”**

### Q2 (Multilingual Version — Query Translation Required)
Ask the same question in a second language of your choice (examples below).  
Your pipeline must still work.

Examples:
- **Malay:** “Cadangkan 2–3 pelajar…”
- **Arabic:** “اقترح ٢–٣ طلاب…”
- **Hindi:** “2–3 छात्रों की टीम सुझाइए…”

(You can write your own translation.)

---

## Requirements (What You Must Implement)
### Retrieval Requirements
You must use:
- **Query decomposition** (break the question into sub-queries like: “NLP/LLM”, “backend/API”, “evaluation/metrics”, “mentor fit”)
- **Multi-query retrieval** per sub-question (generate 2–3 rewrites per sub-query OR at least per main question)
- **Union + deduplication** of retrieved documents
- A final cap (e.g., top 8–12 chunks) to avoid prompt bloat

### Answering Requirements
- Answers must be based **only** on retrieved context
- Must output:
  - Team list (2–3 students)
  - Roles (example roles: Retrieval Engineer, Backend/API Engineer, Evaluation Lead)
  - Mentor name + reason
  - Evidence snippets (2–4 per person/mentor)
- If not supported by context, output exactly:
  - **Not found in the provided data.**

---

## Output Format (Strict)
Return your final answer in the following structure:

1. **Team Recommendation**
   - Student A — Role — Reason (1–2 lines)
   - Student B — Role — Reason (1–2 lines)
   - (Optional) Student C — Role — Reason (1–2 lines)

2. **Mentor Recommendation**
   - Mentor X — Why (1–2 lines)

3. **Evidence**
   - **Student A**
     - Evidence 1
     - Evidence 2
   - **Student B**
     - Evidence 1
     - Evidence 2
   - **Mentor X**
     - Evidence 1
     - Evidence 2

Notes:
- Evidence must be short and directly relevant.
- Evidence must be clearly attributable to retrieved content (not “general knowledge”).

---

## What Makes This Easy + Challenging
### Easy
- You already have indexed data (vector DB ready)
- Multi-query and decomposition patterns exist in your notebooks

### Challenging (the real learning goals)
- Constraints require **coverage across different students**
- Single-query retrieval often misses one constraint → you must decompose
- Multilingual input requires translation before retrieval
- Evidence must be precise (forces grounded generation)

---

## Suggested Approach (High-Level)
1. Translate the question to English (if multilingual input)
2. Decompose into 4 sub-queries:
   - NLP/LLM/RAG candidate(s)
   - Backend/API candidate(s)
   - Evaluation/metrics/data-discipline candidate(s)
   - Mentor fit
3. Generate 2–3 rewrite queries per sub-query
4. Retrieve k=3 per query → union → dedupe → cap to max 10–12 chunks
5. Answer with strict “only from context” rules
6. Include evidence snippets per person

---

## Evaluation Checklist (Self-Check)
Your solution is “correct” if:
- ✅ Team satisfies all constraints (LLM + backend + evaluation preference)
- ✅ Mentor recommendation is grounded in context
- ✅ Evidence is provided (2–4 per person)
- ✅ Multilingual version works
- ✅ If context doesn’t support something, you say “Not found in the provided data.”

---

## Submission
Provide:
- A notebook cell output for **Q1** and **Q2**
- The retrieved query list (show generated sub-queries)
- A short note describing where Simple RAG failed and how Multi-Query + Decomposition fixed it
