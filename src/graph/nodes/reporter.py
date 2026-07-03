import os
import json
from groq import Groq
from dotenv import load_dotenv
from src.graph.state import AgentAnalysisState
from langsmith import traceable

load_dotenv()

REPORTER_SYSTEM_PROMPT = """You are a senior data scientist writing a professional analysis report.

STRICT RULES:
1. Only interpret numbers and results that are explicitly provided to you.
2. Never invent statistics, p-values, or conclusions not present in the input.
3. Write in clear business language.
4. Structure the report with these exact sections:
   ## Executive Summary
   ## Data Overview
   ## Analysis Results
   ## Statistical Findings
   ## Limitations & Confidence
   ## Recommendations
5. For each finding, note its confidence level (High/Medium/Low).
6. If a step failed or hit max retries, explicitly mention this as a limitation.
7. Keep the report concise — aim for 400-600 words total.
"""


def build_reporter_prompt(state: AgentAnalysisState) -> str:
    plan = state["analysis_plan"]
    profile = state["data_profile"]
    business_query = state["business_query"]
    metrics = state.get("structured_metrics_output", {})
    confidence_signals = state.get("confidence_signals", {})
    error_history = state.get("error_history", [])
    artifact_paths = state.get("artifact_paths", [])

    steps_summary = []
    for step in plan:
        step_key = f"step_{step['step_id']}"
        confidence = confidence_signals.get(step_key, "unknown")
        steps_summary.append(
            f"- Step {step['step_id']} [{step['phase']}]: {step['description']} "
            f"| Methodology: {step['selected_methodology']} "
            f"| Confidence: {confidence}"
        )

    failed_steps = []
    for err in error_history:
        failed_steps.append(
            f"- Step {err['step_id']} failed on retry {err['retry']}: "
            f"{err.get('stderr', '')[:150]}"
        )

    metrics_str = json.dumps(metrics, indent=2) if metrics else "No metrics collected."
    artifacts_str = "\n".join(artifact_paths) if artifact_paths else "No artifacts generated."

    return f"""Generate a professional data analysis report for the following:

BUSINESS QUESTION: {business_query}

DATASET OVERVIEW:
- Rows: {profile['row_count']}
- Columns: {profile['column_count']}
- Column names: {profile['columns']}

ANALYSIS STEPS EXECUTED:
{chr(10).join(steps_summary)}

VERIFIED METRICS FROM EXECUTION:
{metrics_str}

GENERATED ARTIFACTS:
{artifacts_str}

FAILED STEPS / ERRORS:
{chr(10).join(failed_steps) if failed_steps else 'None — all steps completed successfully.'}

CONFIDENCE SIGNALS:
{json.dumps(confidence_signals, indent=2)}

Write the complete analysis report now using only the verified data above:"""


@traceable(name="reporter_node", run_type="llm")
def reporter_node(state: AgentAnalysisState) -> dict:
    print("[Reporter] Generating final analysis report...")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = build_reporter_prompt(state)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        temperature=0.3,
        messages=[
            {"role": "system", "content": REPORTER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    report = response.choices[0].message.content.strip()
    print("[Reporter] Report generated successfully.")
    print(f"[Reporter] Report length: {len(report.split())} words")

    return {"final_markdown_report": report}