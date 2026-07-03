"""
Step 18 - DeepEval tests for the Critic node.
Validates that assumption-violation detection (parametric test used on
non-normal data) fires correctly and that retry/graceful-degradation
logic behaves as designed.
"""

from src.graph.nodes.critic import (
    is_parametric_test,
    check_assumption_violation,
    critic_node,
)


def _base_state(**overrides):
    state = {
        "execution_success": True,
        "runtime_stdout": "done",
        "runtime_stderr": "",
        "retry_count": 0,
        "max_retries_per_step": 3,
        "error_history": [],
        "confidence_signals": {},
        "analysis_plan": [
            {
                "step_id": 1,
                "phase": "Hypothesis_Testing",
                "description": "Compare salary across departments",
                "selected_methodology": "One-way ANOVA",
                "mathematical_justification": "Comparing 3+ group means",
            }
        ],
        "current_step_index": 0,
        "data_profile": {
            "datatypes": {"salary": "Float64"},
            "normality_flags": {"salary": False},
        },
    }
    state.update(overrides)
    return state


def test_is_parametric_test_detects_anova():
    assert is_parametric_test("One-way ANOVA") is True


def test_is_parametric_test_rejects_nonparametric():
    assert is_parametric_test("Kruskal-Wallis H Test") is False


def test_violation_detected_for_parametric_on_nonnormal_data():
    state = _base_state()
    violated, feedback = check_assumption_violation(state)
    assert violated is True
    assert "ASSUMPTION VIOLATION" in feedback
    assert "Kruskal-Wallis" in feedback


def test_no_violation_for_nonparametric_method():
    state = _base_state()
    state["analysis_plan"][0]["selected_methodology"] = "Kruskal-Wallis H Test"
    violated, _ = check_assumption_violation(state)
    assert violated is False


def test_critic_node_sends_back_to_coder_on_violation():
    state = _base_state()
    result = critic_node(state)
    assert result["statistical_assumptions_passed"] is False
    assert result["retry_count"] == 1
    assert result["critic_feedback"] is not None


def test_critic_node_advances_step_on_pass():
    state = _base_state()
    state["analysis_plan"][0]["selected_methodology"] = "Kruskal-Wallis H Test"
    result = critic_node(state)
    assert result["statistical_assumptions_passed"] is True
    assert result["execution_success"] is True
    assert result["current_step_index"] == 1


def test_critic_node_graceful_degradation_at_max_retries():
    state = _base_state(retry_count=3, max_retries_per_step=3)
    result = critic_node(state)
    assert result["statistical_assumptions_passed"] is False
    assert result["confidence_signals"]["step_1"] == "low"


def test_critic_node_catches_execution_failure():
    state = _base_state(execution_success=False, runtime_stderr="NameError: df is not defined")
    result = critic_node(state)
    assert result["execution_success"] is False
    assert result["retry_count"] == 1
    assert "EXECUTION ERROR" in result["critic_feedback"]