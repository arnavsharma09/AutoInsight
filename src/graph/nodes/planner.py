import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv
from src.graph.state import AgentAnalysisState, StrategicStep
from langsmith import traceable

load_dotenv()

PLANNING_TOOL = {
    "name": "generate_analysis_plan",
    "description": "Generate a structured step-by-step analysis plan for the given dataset and business question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "List of strategic analysis steps to execute in order",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_id": {"type": "integer"},
                        "phase": {
                            "type": "string",
                            "enum": ["Data_Cleaning", "EDA", "Hypothesis_Testing", "Visualization"]
                        },
                        "description": {"type": "string"},
                        "selected_methodology": {"type": "string"},
                        "mathematical_justification": {"type": "string"}
                    },
                    "required": ["step_id", "phase", "description", "selected_methodology", "mathematical_justification"]
                }
            }
        },
        "required": ["steps"]
    }
}


def build_planner_prompt(profile, router_output: str, business_query: str) -> str:
    numeric_cols = [
        col for col, dtype in profile["datatypes"].items()
        if any(t in dtype for t in ["Int", "Float"])
    ]
    categorical_cols = [
        col for col, dtype in profile["datatypes"].items()
        if any(t in dtype for t in ["Utf8", "String", "Categorical"])
    ]
    normal_cols = [col for col, flag in profile["normality_flags"].items() if flag]
    skewed_cols = [col for col, flag in profile["skewness_flags"].items() if flag]
    high_missing = [
        col for col, ratio in profile["missing_value_ratios"].items()
        if ratio > 0.05
    ]

    return f"""You are a senior data scientist generating a precise analysis plan.

BUSINESS QUESTION: {business_query}

DATASET PROFILE:
- Rows: {profile['row_count']}, Columns: {profile['column_count']}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Normal distributions: {normal_cols if normal_cols else 'None'}
- Skewed columns: {skewed_cols if skewed_cols else 'None'}
- Columns with >5% missing values: {high_missing if high_missing else 'None'}

ROUTER METHODOLOGY DECISION:
{router_output}

HARD CONSTRAINTS - YOU MUST FOLLOW THESE:
1. The Hypothesis_Testing step MUST use the methodology specified in ROUTER METHODOLOGY DECISION above.
2. If any columns have >5% missing values, the FIRST step must be Data_Cleaning.
3. Always include at least one Visualization step as the final step.
4. Generate between 3 and 6 steps total. Do not over-engineer.
5. All code will use Polars (not pandas) and SciPy.
6. Each step must be atomic.

Call the generate_analysis_plan tool with your structured plan now."""


@traceable(name="planner_node", run_type="llm")
def planner_node(state: AgentAnalysisState) -> dict:
    print("[Planner] Generating analysis plan with Anthropic...")

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    profile = state["data_profile"]
    router_output = state.get("router_output", "No routing output available.")
    business_query = state["business_query"]
    prompt = build_planner_prompt(profile, router_output, business_query)

    try:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=2000,
            tools=[PLANNING_TOOL],
            tool_choice={"type": "tool", "name": "generate_analysis_plan"},
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            print("[Planner] Rate limit hit. Waiting 60 seconds...")
            time.sleep(60)
            raise
        raise

    tool_use_block = next(b for b in response.content if b.type == "tool_use")
    plan_data = tool_use_block.input
    steps_raw = plan_data["steps"]

    steps: list[StrategicStep] = [
        StrategicStep(
            step_id=s["step_id"],
            phase=s["phase"],
            description=s["description"],
            selected_methodology=s["selected_methodology"],
            mathematical_justification=s["mathematical_justification"]
        )
        for s in steps_raw
    ]

    print(f"[Planner] Generated {len(steps)} steps:")
    for step in steps:
        print(f"  Step {step['step_id']} [{step['phase']}]: {step['description'][:60]}...")

    return {
        "analysis_plan": steps,
        "plan_approved_by_human": False,
        "current_step_index": 0,
    }
