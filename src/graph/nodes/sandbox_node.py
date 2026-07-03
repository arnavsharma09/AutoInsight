from src.graph.state import AgentAnalysisState
from src.sandbox.security import check_code_safety
from src.sandbox.kernel import execute_code


def sandbox_node(state: AgentAnalysisState) -> dict:
    code = state.get("active_code_block", "")
    current_step = state["analysis_plan"][state["current_step_index"]]

    print(f"[Sandbox] Running code for step {current_step['step_id']}...")

    if not code or not code.strip():
        return {
            "execution_success": False,
            "runtime_stdout": "",
            "runtime_stderr": "No code was generated.",
            "structured_metrics_output": {},
        }

    is_safe, violations = check_code_safety(code)
    if not is_safe:
        violation_str = "\n".join(violations)
        print(f"[Sandbox] Security check FAILED:\n{violation_str}")
        return {
            "execution_success": False,
            "runtime_stdout": "",
            "runtime_stderr": f"Security violation:\n{violation_str}",
            "structured_metrics_output": {},
        }

    print(f"[Sandbox] Security check passed. Executing...")
    result = execute_code(code)

    print(f"[Sandbox] Execution complete. Success: {result['execution_success']}")
    if result["stdout"]:
        print(f"[Sandbox] stdout: {result['stdout'][:200]}")
    if result["stderr"]:
        print(f"[Sandbox] stderr: {result['stderr'][:3000]}")

    artifact_paths = list(state.get("artifact_paths", []))
    artifact_paths.extend(result.get("artifacts", []))

    return {
        "execution_success": result["execution_success"],
        "runtime_stdout": result["stdout"],
        "runtime_stderr": result["stderr"],
        "structured_metrics_output": result["metrics"],
        "artifact_paths": artifact_paths,
    }