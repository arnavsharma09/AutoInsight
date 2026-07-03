from langgraph.graph import StateGraph, END
from src.graph.state import AgentAnalysisState
from src.graph.nodes.profiler import profiler_node
from src.graph.nodes.router import router_node
from src.graph.nodes.planner import planner_node
from src.graph.nodes.coder import coder_node
from src.graph.nodes.sandbox_node import sandbox_node
from src.graph.nodes.critic import critic_node
from src.graph.nodes.reporter import reporter_node


def critic_routing(state: AgentAnalysisState) -> str:
    if not state["execution_success"]:
        if state["retry_count"] < state["max_retries_per_step"]:
            return "coder"
        else:
            return "reporter"

    if not state["statistical_assumptions_passed"]:
        if state["retry_count"] < state["max_retries_per_step"]:
            return "coder"
        else:
            return "reporter"

    plan = state.get("analysis_plan", [])
    current = state.get("current_step_index", 0)
    if current >= len(plan) - 1:
        return "reporter"

    return "coder"


def build_graph():
    builder = StateGraph(AgentAnalysisState)

    builder.add_node("profiler", profiler_node)
    builder.add_node("router", router_node)
    builder.add_node("planner", planner_node)
    builder.add_node("coder", coder_node)
    builder.add_node("sandbox", sandbox_node)
    builder.add_node("critic", critic_node)
    builder.add_node("reporter", reporter_node)

    builder.set_entry_point("profiler")
    builder.add_edge("profiler", "router")
    builder.add_edge("router", "planner")
    builder.add_edge("planner", "coder")
    builder.add_edge("coder", "sandbox")
    builder.add_edge("sandbox", "critic")

    builder.add_conditional_edges(
        "critic",
        critic_routing,
        {
            "coder": "coder",
            "reporter": "reporter",
        }
    )

    builder.add_edge("reporter", END)

    return builder.compile(interrupt_before=["coder"])


graph = build_graph()