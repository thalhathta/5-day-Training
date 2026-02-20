# Problem 1 — Top-K Student Ranking (Strict JSON + Verbatim Evidence)

## Context
You are provided with a text file named **`students_details.txt`** that contains multiple student profiles (names, backgrounds, skills, projects, etc.).  
Your task is to build a **LangChain-based ranking tool** that selects the **Top-K best candidates** for a given requirement and returns the result as **strict JSON** with **verbatim evidence** copied from the file.

This problem is designed to surface the limitations of “paste-all-text prompting” and naturally motivate the next session on **RAG**.

---

## Objective
Given a query describing a target role, rank students from `students_details.txt` and return:
- The **Top 3** candidates (**K = 3**)
- A numeric **score** for each selected candidate
- Clear **reasons** for selection
- **Evidence**: exact phrases copied from the file (no paraphrasing)
- A small list of **not selected** candidates with brief reasons

---

## Inputs
### 1) Dataset
- `students_details.txt` (text only)

### 2) Query (use exactly this)
> **Select the top 3 students best suited to build a RAG-based academic assistant using LLMs. Prioritize experience in NLP/LLMs, Python, vector databases, LangChain, APIs, or backend integration.**

### 3) K
- `K = 3`

---

## Required Output (STRICT JSON ONLY)
Your final output must be **valid JSON** and match this structure **exactly**:

```json
{
  "query": "string",
  "top_k": [
    {
      "name": "string",
      "score": 0,
      "reasons": ["string"],
      "evidence": ["string"]
    }
  ],
  "not_selected": [
    {
      "name": "string",
      "reason": "string"
    }
  ]
}

