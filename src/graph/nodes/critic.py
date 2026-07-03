from src.graph.state import AgentAnalysisState

PARAMETRIC_TESTS = {
    "t-test", "t_test", "ttest", "independent samples t-test",
    "paired samples t-test", "welch t-test", "one-way anova",
    "anova", "two-way anova", "repeated measures anova",
    "pearson correlation", "pearson",
    "ordinary least squares", "ols regression",
}

NON_PARAMETRIC_EQUIVALENTS = {
    "independent samples t-test": "Mann-Whitney U Test",
    "welch t-test": "Mann-Whitney U Test",
    "t-test": "Mann-Whitney U Test",
    "one-way anova": "Kruskal-Wallis H Test with Dunn Post-Hoc",
    "anova": "Kruskal-Wallis H Test with Dunn Post-Hoc",
    "pearson correlation": "Spearman Rank Correlation",
    "pearson": "Spearman Rank Correlation",
    "ordinary least squares": "Log-Transformed OLS or Quantile Regression",
    "ols regression": "Log-Transformed OLS or Quantile Regression",
}


def is_parametric_test(methodology: str) -> bool:
    methodology_lower = methodology.lower()
    return any(test in methodology_lower for test in PARAMETRIC_TESTS)


def check_assumption_violation(state: AgentAnalysisState) -> tuple[bool, str]:
    plan = state["analysis_plan"]
    current_index = state["current_step_index"]
    current_step = plan[current_index]
    methodology = current_step["selected_methodology"]

    if not is_parametric_test(methodology):
        return False, ""

    profile = state["data_profile"]
    normality_flags = profile.get("normality_flags", {})
    numeric_normality = {
        col: flag for col, flag in normality_flags.items()
        if profile["datatypes"].get(col, "") not in ("Utf8", "String", "Categorical")
    }

    if not numeric_normality:
        return False, ""

    non_normal_cols = [col for col, flag in numeric_normality.items() if not flag]

    if non_normal_cols:
        methodology_lower = methodology.lower()
        equivalent = None
        for key, val in NON_PARAMETRIC_EQUIVALENTS.items():
            if key in methodology_lower:
                equivalent = val
                break

        feedback = (
            f"ASSUMPTION VIOLATION: Parametric test '{methodology}' was used, "
            f"but Shapiro-Wilk normality test failed for columns: {non_normal_cols}. "
            f"Switch to non-parametric equivalent: '{equivalent or 'a non-parametric test'}'. "
            f"Rewrite the code using the non-parametric method."
        )
        return True, feedback

    return False, ""


def critic_node(state: AgentAnalysisState) -> dict:
    print("[Critic] Running validation rules...")

    execution_success = state.get("execution_success", False)
    runtime_stderr = state.get("runtime_stderr", "")
    runtime_stdout = state.get("runtime_stdout", "")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries_per_step", 3)
    error_history = state.get("error_history", [])
    plan = state["analysis_plan"]
    current_index = state["current_step_index"]
    current_step = plan[current_index]

    if retry_count >= max_retries:
        print(f"[Critic] Rule C: Max retries ({max_retries}) reached. Graceful degradation.")
        confidence_signals = dict(state.get("confidence_signals", {}))
        confidence_signals[f"step_{current_step['step_id']}"] = "low"
        return {
            "statistical_assumptions_passed": False,
            "critic_feedback": f"Max retries exceeded for step {current_step['step_id']}.",
            "confidence_signals": confidence_signals,
        }

    if not execution_success or (runtime_stderr and runtime_stderr.strip()):
        print(f"[Critic] Rule A: Execution failed for step {current_step['step_id']}.")
        error_entry = {
            "step_id": current_step["step_id"],
            "retry": retry_count,
            "stderr": runtime_stderr,
            "stdout": runtime_stdout,
        }
        updated_error_history = error_history + [error_entry]
        feedback = (
            f"EXECUTION ERROR in step {current_step['step_id']}:\n"
            f"stderr: {runtime_stderr[:1000]}\n"
            f"Fix the code so it runs without errors."
        )
        print(f"[Critic] Sending back to Coder. Retry {retry_count + 1}/{max_retries}")
        return {
            "execution_success": False,
            "statistical_assumptions_passed": False,
            "critic_feedback": feedback,
            "retry_count": retry_count + 1,
            "error_history": updated_error_history,
        }

    violated, feedback = check_assumption_violation(state)
    if violated:
        print(f"[Critic] Rule B: Statistical assumption violation.")
        error_entry = {
            "step_id": current_step["step_id"],
            "retry": retry_count,
            "stderr": "Assumption violation",
            "stdout": runtime_stdout,
        }
        updated_error_history = error_history + [error_entry]
        print(f"[Critic] Sending back to Coder. Retry {retry_count + 1}/{max_retries}")
        return {
            "statistical_assumptions_passed": False,
            "critic_feedback": feedback,
            "retry_count": retry_count + 1,
            "error_history": updated_error_history,
        }

    print(f"[Critic] Step {current_step['step_id']} passed all validation rules.")
    confidence_signals = dict(state.get("confidence_signals", {}))
    if retry_count == 0:
        confidence_signals[f"step_{current_step['step_id']}"] = "high"
    else:
        confidence_signals[f"step_{current_step['step_id']}"] = "medium"

    return {
        "execution_success": True,
        "statistical_assumptions_passed": True,
        "critic_feedback": None,
        "retry_count": 0,
        "current_step_index": current_index + 1,
        "confidence_signals": confidence_signals,
    }