import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AutoInsight",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AutoInsight")
st.caption("Autonomous Multi-Agent Data Analysis System")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "status" not in st.session_state:
    st.session_state.status = None
if "plan" not in st.session_state:
    st.session_state.plan = None
if "report" not in st.session_state:
    st.session_state.report = None


def get_status():
    if not st.session_state.session_id:
        return None
    try:
        r = requests.get(f"{API_URL}/status/{st.session_state.session_id}", timeout=10)
        return r.json()
    except Exception:
        return None


def reset():
    st.session_state.session_id = None
    st.session_state.status = None
    st.session_state.plan = None
    st.session_state.report = None


# --- Phase 1: Upload ---
if st.session_state.status is None:
    st.subheader("Step 1 — Upload your dataset")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    business_query = st.text_input(
        "Business question",
        placeholder="e.g. Is there a significant difference in salary between departments?"
    )

    if st.button("Start Analysis", type="primary", disabled=not (uploaded_file and business_query)):
        with st.spinner("Uploading file..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            r = requests.post(f"{API_URL}/upload", files=files, timeout=30)
            if r.status_code != 200:
                st.error(f"Upload failed: {r.text}")
                st.stop()
            session_id = r.json()["session_id"]
            st.session_state.session_id = session_id

        with st.spinner("Starting analysis..."):
            r = requests.post(
                f"{API_URL}/analyze?session_id={session_id}",
                json={"business_query": business_query},
                timeout=30
            )
            if r.status_code != 200:
                st.error(f"Analysis failed to start: {r.text}")
                st.stop()
            st.session_state.status = "started"
            st.rerun()


# --- Phase 2: Waiting for plan / Plan approval ---
elif st.session_state.status in ("started", "profiling", "planning", "awaiting_approval"):
    status_data = get_status()
    current_status = status_data.get("status") if status_data else st.session_state.status
    st.session_state.status = current_status

    if current_status in ("started", "profiling", "planning"):
        status_labels = {
            "started": "🔄 Initializing...",
            "profiling": "📊 Profiling dataset...",
            "planning": "🧠 Generating analysis plan...",
        }
        st.info(status_labels.get(current_status, "Processing..."))
        with st.spinner("Please wait..."):
            time.sleep(3)
            st.rerun()

    elif current_status == "awaiting_approval":
        plan = status_data.get("analysis_plan", [])
        profile = status_data.get("data_profile", {})
        st.session_state.plan = plan

        st.subheader("Step 2 — Review the analysis plan")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", profile.get("row_count", "—"))
        col2.metric("Columns", profile.get("column_count", "—"))
        col3.metric("Analysis Steps", len(plan))

        st.markdown("---")
        st.markdown("**Proposed Analysis Steps:**")

        for step in plan:
            with st.expander(f"Step {step['step_id']} — {step['phase']}: {step['description'][:60]}..."):
                st.markdown(f"**Description:** {step['description']}")
                st.markdown(f"**Methodology:** {step['selected_methodology']}")
                st.markdown(f"**Justification:** {step['mathematical_justification']}")

        st.markdown("---")
        col_approve, col_cancel = st.columns([1, 4])

        with col_approve:
            if st.button("✅ Approve Plan", type="primary"):
                r = requests.post(
                    f"{API_URL}/approve",
                    json={"session_id": st.session_state.session_id},
                    timeout=10
                )
                if r.status_code == 200:
                    st.session_state.status = "running"
                    st.rerun()
                else:
                    st.error(f"Approval failed: {r.text}")

        with col_cancel:
            if st.button("❌ Cancel"):
                reset()
                st.rerun()


# --- Phase 3: Running ---
elif st.session_state.status in ("running", "reporting"):
    status_data = get_status()
    current_status = status_data.get("status") if status_data else st.session_state.status
    st.session_state.status = current_status

    if current_status in ("running", "reporting"):
        status_labels = {
            "running": "⚙️ Running analysis steps...",
            "reporting": "📝 Generating report...",
        }
        st.info(status_labels.get(current_status, "Processing..."))

        if st.session_state.plan:
            st.markdown("**Analysis steps:**")
            for step in st.session_state.plan:
                st.markdown(f"- Step {step['step_id']} [{step['phase']}]: {step['description'][:70]}...")

        with st.spinner("Please wait — this takes 1-2 minutes..."):
            time.sleep(3)
            st.rerun()

    elif current_status == "complete":
        st.session_state.status = "complete"
        st.rerun()

    elif current_status == "error":
        st.error(f"Analysis failed: {status_data.get('error')}")
        if st.button("Start over"):
            reset()
            st.rerun()


# --- Phase 4: Complete ---
elif st.session_state.status == "complete":
    st.success("✅ Analysis complete!")

    r = requests.get(
        f"{API_URL}/report/{st.session_state.session_id}",
        timeout=10
    )

    if r.status_code == 200:
        data = r.json()
        report = data.get("report", "")
        confidence = data.get("confidence_signals", {})
        st.session_state.report = report

        tab1, tab2 = st.tabs(["📄 Report", "📊 Confidence Signals"])

        with tab1:
            st.markdown(report)
            st.download_button(
                label="⬇️ Download Report",
                data=report,
                file_name="autoinsight_report.md",
                mime="text/markdown"
            )

        with tab2:
            if confidence:
                for step, level in confidence.items():
                    color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(level, "⚪")
                    st.markdown(f"{color} **{step}**: {level.upper()}")
            else:
                st.info("No confidence signals available.")

    else:
        st.error("Failed to fetch report.")

    if st.button("🔄 Start new analysis"):
        reset()
        st.rerun()