from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict


class DataProfile(TypedDict):
    row_count: int
    column_count: int
    columns: List[str]
    datatypes: Dict[str, str]
    missing_value_ratios: Dict[str, float]
    cardinality: Dict[str, int]
    skewness_flags: Dict[str, bool]
    normality_flags: Dict[str, bool]


class StrategicStep(TypedDict):
    step_id: int
    phase: str
    description: str
    selected_methodology: str
    mathematical_justification: str


class AgentAnalysisState(TypedDict):
    # Input
    business_query: str
    raw_data_path: str

    # Profiler output
    clean_data_path: Optional[str]
    data_profile: Optional[DataProfile]

    # Router output
    router_output: Optional[str]

    # Planner output
    analysis_plan: List[StrategicStep]
    plan_approved_by_human: bool
    current_step_index: int

    # Coder output
    active_code_block: Optional[str]

    # Sandbox output
    execution_success: bool
    runtime_stdout: Optional[str]
    runtime_stderr: Optional[str]
    structured_metrics_output: Optional[Dict[str, Any]]

    # Critic output
    statistical_assumptions_passed: bool
    critic_feedback: Optional[str]
    retry_count: int
    max_retries_per_step: int
    error_history: List[Dict[str, Any]]

    # Shared artifacts
    artifact_paths: List[str]
    confidence_signals: Dict[str, str]

    # Reporter output
    final_markdown_report: Optional[str]