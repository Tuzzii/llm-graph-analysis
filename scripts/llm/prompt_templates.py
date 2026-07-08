# ──────────────────────────────────────────────
# ZERO-SHOT PROMPT
# (được dùng cho tất cả task: SP, MF, Connectivity, BM)
# ──────────────────────────────────────────────
def zero_shot_prompt(question):
    return f"""You are an expert graph algorithm solver.
Please read the following graph description and answer the query.

Question:
{question}

Provide only the final answer.
"""

# ──────────────────────────────────────────────
# CHAIN-OF-THOUGHT PROMPT (CoT)
# ──────────────────────────────────────────────
def cot_prompt(question):
    return f"""You are an expert graph algorithm solver.
Use detailed step-by-step reasoning to solve the problem.

Question:
{question}

Explain your reasoning step-by-step before giving the final answer.
"""

# ──────────────────────────────────────────────
# CoT WITHOUT REVEAL (for paper-style evaluation)
# ──────────────────────────────────────────────
def cot_hidden_prompt(question):
    return f"""You are an expert graph algorithm solver.
Think step-by-step, but DO NOT reveal your reasoning.

Question:
{question}

Think through the solution internally, then provide only the final answer.
"""
