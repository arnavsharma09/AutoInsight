import re
import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv
from src.graph.state import AgentAnalysisState
from langsmith import traceable

load_dotenv()

SYSTEM_PROMPT = """You are an expert Python data scientist writing executable analysis code.

STRICT RULES:
1. Use Polars (imported as pl) for ALL data operations. Never use pandas.
2. Use SciPy (scipy.stats) for statistical tests.
3. Write ALL metrics (p-values, statistics, confidence intervals) to /workspace/metrics.json
4. Save ALL plots as HTML files to /workspace/artifacts/ using plotly
5. The dataset is already loaded - use the variable `df` which is a Polars DataFrame
6. Return ONLY raw Python code. No markdown, no backticks, no explanations.
7. Every numeric result must be JSON serializable (convert numpy types to float/int)
8. You MUST write metrics.json using: open('/workspace/metrics.json', 'w')
9. This code runs inside a secure Docker container where open() is fully allowed.
10. Do NOT use hasattr() anywhere in your code.
11. Keep code concise - maximum 60 lines per step.

AVAILABLE APIS:
- polars: pl.read_csv(), df.filter(), df.group_by(), df.select(), df['col'].to_numpy()
- scipy.stats.kruskal(*arrays) -> (statistic, pvalue)
- scipy.stats.mannwhitneyu(x, y) -> (statistic, pvalue)
- scipy.stats.shapiro(x) -> (statistic, pvalue)
- scipy.stats.spearmanr(x, y) -> (correlation, pvalue)
- scipy.stats.pearsonr(x, y) -> (correlation, pvalue)
- statsmodels.stats.multicomp.pairwise_tukeyhsd(endog, groups)
- plotly.express as px: px.box(), px.violin(), px.scatter(), px.histogram()
- scikit_posthocs: import scikit_posthocs as sp; sp.posthoc_dunn(df, val_col, group_col)

ONLY USE THESE IMPORTS — no other packages:
import polars as pl
import numpy as np
import json
from scipy import stats
import plotly.express as px
import scikit_posthocs as sp
import statsmodels.stats.multicomp as mc

METRICS JSON FORMAT:
import json
metrics = {"key": value}
with open('/workspace/metrics.json', 'w') as f:
    json.dump(metrics, f)
"""


def clean_code_output(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```python\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def build_coder_prompt(state: AgentAnalysisState) -> str:
    plan = state["analysis_plan"]
    current_index = state["current_step_index"]
    current_step = plan[current_index]

    data_path = state.get("clean_data_path") or state["raw_data_path"]
    profile = state["data_profile"]
    columns = profile["columns"]
    datatypes = profile["datatypes"]

    critic_feedback = state.get("critic_feedback")
    error_history = state.get("error_history", [])
    retry_count = state.get("retry_count", 0)

    prompt = f"""Write Python code for this analysis step:

STEP {current_step['step_id']}: {current_step['phase']}
Description: {current_step['description']}
Methodology: {current_step['selected_methodology']}
Justification: {current_step['mathematical_justification']}

DATASET INFO:
- File path: {data_path}
- Columns: {columns}
- Dtypes: {datatypes}

IMPORTANT: The variable `df` is already loaded as a Polars DataFrame from a previous step.
If this is step 1, load it yourself: df = pl.read_csv('{data_path}')

REMINDER: You MUST save metrics to /workspace/metrics.json using open().
open() is fully allowed in this Docker environment.
Keep code under 60 lines. Only use the approved imports listed in your instructions.
"""

    if retry_count > 0 and critic_feedback:
        prompt += f"""
THIS IS RETRY ATTEMPT {retry_count}.
YOUR PREVIOUS CODE FAILED. Here is the feedback:

CRITIC FEEDBACK: {critic_feedback}
"""
        if error_history:
            last_error = error_history[-1]
            prompt += f"""
LAST ERROR:
- stderr: {last_error.get('stderr', 'None')[:500]}
- stdout: {last_error.get('stdout', 'None')[:200]}

Fix the exact issue described. Do not repeat the same mistake.
"""

    prompt += "\nWrite the complete Python code now. Raw code only, no explanations:"
    return prompt


@traceable(name="coder_node", run_type="llm")
def coder_node(state: AgentAnalysisState) -> dict:
    plan = state["analysis_plan"]
    current_index = state["current_step_index"]
    current_step = plan[current_index]

    print(f"[Coder] Generating code for Step {current_step['step_id']}: {current_step['phase']}")
    if state.get("retry_count", 0) > 0:
        print(f"[Coder] Retry attempt {state['retry_count']} — applying critic feedback")

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = build_coder_prompt(state)

    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            print("[Coder] Rate limit hit. Waiting 60 seconds...")
            time.sleep(60)
            raise
        raise

    raw_code = response.content[0].text
    clean_code = clean_code_output(raw_code)

    print(f"[Coder] Generated {len(clean_code.splitlines())} lines of code")
    return {"active_code_block": clean_code}
