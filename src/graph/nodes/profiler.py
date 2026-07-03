import polars as pl
import numpy as np
from scipy import stats
from src.graph.state import AgentAnalysisState, DataProfile


NORMALITY_SAMPLE_SIZE = 5000
SKEWNESS_THRESHOLD = 1.0
SHAPIRO_P_THRESHOLD = 0.05


def run_profiler(data_path: str) -> DataProfile:
    lazy_df = pl.scan_csv(data_path)
    schema = lazy_df.collect_schema()

    df = lazy_df.collect()

    row_count = df.height
    column_count = df.width
    columns = df.columns

    datatypes = {col: str(schema[col]) for col in columns}

    missing_value_ratios = {
        col: round(df[col].is_null().sum() / row_count, 4)
        for col in columns
    }

    cardinality = {
        col: df[col].n_unique()
        for col in columns
    }

    numeric_cols = [
        col for col in columns
        if schema[col] in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8)
    ]

    skewness_flags = {}
    for col in columns:
        if col in numeric_cols:
            series = df[col].drop_nulls().to_numpy()
            if len(series) > 1:
                skew = float(stats.skew(series))
                skewness_flags[col] = abs(skew) > SKEWNESS_THRESHOLD
            else:
                skewness_flags[col] = False
        else:
            skewness_flags[col] = False

    normality_flags = {}
    for col in columns:
        if col in numeric_cols:
            series = df[col].drop_nulls().to_numpy()
            if len(series) < 3:
                normality_flags[col] = False
                continue
            if len(series) > NORMALITY_SAMPLE_SIZE:
                rng = np.random.default_rng(42)
                series = rng.choice(series, size=NORMALITY_SAMPLE_SIZE, replace=False)
            _, p_value = stats.shapiro(series)
            normality_flags[col] = p_value >= SHAPIRO_P_THRESHOLD
        else:
            normality_flags[col] = False

    return DataProfile(
        row_count=row_count,
        column_count=column_count,
        columns=columns,
        datatypes=datatypes,
        missing_value_ratios=missing_value_ratios,
        cardinality=cardinality,
        skewness_flags=skewness_flags,
        normality_flags=normality_flags,
    )


def profiler_node(state: AgentAnalysisState) -> dict:
    print(f"[Profiler] Scanning: {state['raw_data_path']}")
    profile = run_profiler(state["raw_data_path"])
    print(f"[Profiler] Done — {profile['row_count']} rows, {profile['column_count']} columns")
    return {"data_profile": profile}