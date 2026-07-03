"""
Step 18 - DeepEval tests for the Router node.
Validates that the hybrid JSON/ChromaDB router selects sound methodologies
for known query + data-profile combinations. These are deterministic
assertions (no LLM judge needed) since router logic is rule-based.
"""

import pytest
from src.graph.nodes.router import (
    detect_query_intent,
    check_normality,
    router_node,
)


def _mock_profile(normal=True, skewed=False):
    return {
        "row_count": 500,
        "column_count": 3,
        "columns": ["department", "salary", "years_experience"],
        "datatypes": {
            "department": "Utf8",
            "salary": "Float64",
            "years_experience": "Int64",
        },
        "missing_value_ratios": {"department": 0.0, "salary": 0.0, "years_experience": 0.0},
        "cardinality": {"department": 4, "salary": 480, "years_experience": 30},
        "skewness_flags": {"salary": skewed, "years_experience": skewed},
        "normality_flags": {"salary": normal, "years_experience": normal},
    }


def test_intent_detection_comparison():
    assert detect_query_intent("Compare salaries between departments") == "comparison"


def test_intent_detection_correlation():
    assert detect_query_intent("Is there a relationship between experience and salary?") == "correlation"


def test_intent_detection_regression():
    assert detect_query_intent("Predict salary based on experience") == "regression"


def test_intent_detection_time_series():
    assert detect_query_intent("What is the monthly trend in revenue?") == "time_series"


def test_normality_check_true_when_all_normal():
    profile = _mock_profile(normal=True)
    assert check_normality(profile) is True


def test_normality_check_false_when_any_non_normal():
    profile = _mock_profile(normal=False)
    assert check_normality(profile) is False


def test_router_selects_nonparametric_when_data_non_normal():
    """
    Critical correctness test: if data is non-normal, router output must
    NOT recommend a parametric test (t-test/ANOVA/Pearson) as-is.
    """
    profile = _mock_profile(normal=False)
    state = {
        "data_profile": profile,
        "business_query": "Compare salaries between departments",
    }
    result = router_node(state)
    output = result["router_output"].lower()

    assert "non-normal" in output or "kruskal" in output or "mann-whitney" in output, (
        f"Router did not account for non-normal data: {output}"
    )


def test_router_output_contains_required_fields():
    profile = _mock_profile(normal=True)
    state = {
        "data_profile": profile,
        "business_query": "Compare salaries between departments",
    }
    result = router_node(state)
    output = result["router_output"]

    assert "SELECTED METHODOLOGY" in output
    assert "JUSTIFICATION" in output
    assert "ROUTING SOURCE" in output