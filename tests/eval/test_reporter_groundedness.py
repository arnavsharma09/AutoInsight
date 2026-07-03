"""
Step 18 - DeepEval faithfulness test for the Reporter node.

Checks that the Reporter LLM doesn't assert facts/numbers not grounded
in the verified metrics context we gave it. Uses DeepEval's
FaithfulnessMetric, which defaults to an OpenAI judge model - the test
auto-skips if OPENAI_API_KEY isn't set, so your suite stays green
without it.
"""

import os
import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.graph.nodes.reporter import reporter_node


def _mock_state():
    return {
        "business_query": "Is there a significant difference in salary between departments?",
        "analysis_plan": [
            {
                "step_id": 1,
                "phase": "Hypothesis_Testing",
                "description": "Compare salary across departments",
                "selected_methodology": "Kruskal-Wallis H Test",
                "mathematical_justification": "Non-normal data across 4 groups",
            }
        ],
        "data_profile": {
            "row_count": 500,
            "column_count": 3,
            "columns": ["department", "salary", "years_experience"],
        },
        "structured_metrics_output": {
            "kruskal_statistic": 14.32,
            "kruskal_pvalue": 0.0025,
            "group_medians": {"Engineering": 95000, "Sales": 72000, "HR": 68000, "Support": 65000},
        },
        "confidence_signals": {"step_1": "high"},
        "error_history": [],
        "artifact_paths": ["/workspace/artifacts/salary_boxplot.html"],
    }


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="FaithfulnessMetric requires OPENAI_API_KEY for the default judge model",
)
def test_reporter_is_faithful_to_metrics():
    state = _mock_state()
    result = reporter_node(state)
    report_text = result["final_markdown_report"]

    retrieval_context = [
        f"kruskal_statistic: {state['structured_metrics_output']['kruskal_statistic']}",
        f"kruskal_pvalue: {state['structured_metrics_output']['kruskal_pvalue']}",
        f"group_medians: {state['structured_metrics_output']['group_medians']}",
        f"row_count: {state['data_profile']['row_count']}",
    ]

    test_case = LLMTestCase(
        input=state["business_query"],
        actual_output=report_text,
        retrieval_context=retrieval_context,
    )

    faithfulness = FaithfulnessMetric(threshold=0.8)
    assert_test(test_case, [faithfulness])


def test_reporter_contains_required_sections():
    """Deterministic structural check - no LLM judge needed."""
    state = _mock_state()
    result = reporter_node(state)
    report = result["final_markdown_report"]

    required_sections = [
        "Executive Summary",
        "Data Overview",
        "Analysis Results",
        "Statistical Findings",
        "Limitations",
        "Recommendations",
    ]
    for section in required_sections:
        assert section in report, f"Missing section: {section}"


def test_reporter_flags_low_confidence_steps():
    """If a step degraded to low confidence, the report must mention it as a limitation."""
    state = _mock_state()
    state["confidence_signals"] = {"step_1": "low"}
    state["error_history"] = [{"step_id": 1, "retry": 3, "stderr": "shape mismatch", "stdout": ""}]

    result = reporter_node(state)
    report_lower = result["final_markdown_report"].lower()

    assert "limitation" in report_lower or "low confidence" in report_lower or "failed" in report_lower