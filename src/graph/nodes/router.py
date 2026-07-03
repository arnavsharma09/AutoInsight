import json
import re
from src.graph.state import AgentAnalysisState
from src.registry.chroma_loader import query_methodologies

RULES_PATH = "src/registry/statistical_rules.json"

COMPARISON_KEYWORDS = ["compare", "difference", "between", "versus", "vs", "group"]
CORRELATION_KEYWORDS = ["correlat", "relationship", "associat", "related", "link"]
REGRESSION_KEYWORDS = ["predict", "regression", "model", "effect of", "impact of", "influence"]
DISTRIBUTION_KEYWORDS = ["distribut", "spread", "skew", "outlier", "normal"]
TIME_KEYWORDS = ["trend", "over time", "time series", "monthly", "yearly", "growth"]
ASSOCIATION_KEYWORDS = ["categorical", "chi", "independence", "association"]


def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["rules"]


def detect_query_intent(business_query: str) -> str:
    q = business_query.lower()
    if any(k in q for k in TIME_KEYWORDS):
        return "time_series"
    if any(k in q for k in REGRESSION_KEYWORDS):
        return "regression"
    if any(k in q for k in CORRELATION_KEYWORDS):
        return "correlation"
    if any(k in q for k in ASSOCIATION_KEYWORDS):
        return "association"
    if any(k in q for k in COMPARISON_KEYWORDS):
        return "comparison"
    if any(k in q for k in DISTRIBUTION_KEYWORDS):
        return "distribution"
    return "unknown"


def count_comparison_groups(profile, business_query: str) -> int:
    q = business_query.lower()
    categorical_cols = [
        col for col, dtype in profile["datatypes"].items()
        if "str" in dtype.lower() or "utf" in dtype.lower()
    ]
    for col in categorical_cols:
        cardinality = profile["cardinality"].get(col, 0)
        if cardinality >= 2:
            return cardinality
    if "two" in q or " 2 " in q:
        return 2
    if "three" in q or " 3 " in q:
        return 3
    return 2


def check_normality(profile) -> bool:
    numeric_normality = {
        col: flag
        for col, flag in profile["normality_flags"].items()
        if profile["datatypes"].get(col, "") not in ("Utf8", "String", "Categorical")
    }
    if not numeric_normality:
        return False
    return all(numeric_normality.values())


def check_target_skewed(profile) -> bool:
    skew_flags = profile["skewness_flags"]
    if not skew_flags:
        return False
    skewed_count = sum(1 for v in skew_flags.values() if v)
    return skewed_count > len(skew_flags) * 0.5


def try_json_routing(profile, business_query: str, rules: list):
    intent = detect_query_intent(business_query)
    all_normal = check_normality(profile)
    group_count = count_comparison_groups(profile, business_query)
    target_skewed = check_target_skewed(profile)

    best_rule = None
    best_score = 0

    for rule in rules:
        score = 0
        conditions = rule["conditions"]

        if intent == "comparison" and "comparison_groups" in conditions:
            score += 2
            if conditions["comparison_groups"] == 2 and group_count == 2:
                score += 2
            elif conditions["comparison_groups"] == "3+" and group_count >= 3:
                score += 2
            if conditions.get("all_groups_normal") == all_normal:
                score += 3

        elif intent == "correlation" and conditions.get("analysis_type") == "correlation":
            score += 3
            if conditions.get("all_groups_normal") == all_normal:
                score += 2

        elif intent == "regression" and conditions.get("analysis_type") == "regression":
            score += 3
            if conditions.get("target_skewed") == target_skewed:
                score += 2

        elif intent == "association" and conditions.get("analysis_type") == "association":
            score += 3

        elif intent == "distribution" and conditions.get("analysis_type") == "distribution":
            score += 3

        elif intent == "time_series" and conditions.get("analysis_type") == "time_series":
            score += 3

        if score > best_score:
            best_score = score
            best_rule = rule

    if best_rule and best_score >= 4:
        return best_rule, best_score
    return None, 0


def router_node(state: AgentAnalysisState) -> dict:
    print("[Router] Starting hybrid methodology routing...")

    profile = state["data_profile"]
    business_query = state["business_query"]
    rules = load_rules()

    matched_rule, score = try_json_routing(profile, business_query, rules)

    if matched_rule:
        methodology = matched_rule["selected_methodology"]
        justification = matched_rule["mathematical_justification"]
        source = "json_ontology"
        print(f"[Router] JSON match (score={score}): {methodology}")
    else:
        print("[Router] No confident JSON match — falling back to ChromaDB...")
        chroma_results = query_methodologies(business_query, n_results=3)
        methodology = chroma_results[0]
        justification = f"Selected via semantic similarity. Top matches: {chroma_results}"
        source = "chromadb_semantic"
        print(f"[Router] ChromaDB result: {methodology}")

    routing_output = (
        f"SELECTED METHODOLOGY: {methodology}\n"
        f"JUSTIFICATION: {justification}\n"
        f"ROUTING SOURCE: {source}\n"
        f"NORMALITY CHECK: {'All numeric columns normal' if check_normality(profile) else 'Non-normal data detected'}\n"
        f"QUERY INTENT: {detect_query_intent(business_query)}"
    )

    print(f"[Router] Done. Routing source: {source}")
    return {"router_output": routing_output}