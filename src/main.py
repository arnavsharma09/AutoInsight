import uuid
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from src.utils.tracing import setup_langsmith
setup_langsmith()

app = FastAPI(
    title="AutoInsight API",
    description="Autonomous multi-agent data analysis system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict = {}

UPLOAD_DIR = Path("workspace/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SANDBOX_CONTAINER = "autoinsight-sandbox"


def copy_file_to_sandbox(local_path: str, sandbox_path: str):
    try:
        subprocess.run(
            ["docker", "cp", local_path, f"{SANDBOX_CONTAINER}:{sandbox_path}"],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to copy file to sandbox: {e.stderr.decode()}")


def run_graph_sync(session_id: str):
    from src.graph.nodes.profiler import run_profiler
    from src.graph.nodes.router import router_node
    from src.graph.nodes.planner import planner_node

    session = sessions[session_id]
    local_path = session["local_path"]
    sandbox_path = session["sandbox_path"]
    business_query = session["business_query"]

    try:
        sessions[session_id]["status"] = "profiling"
        profile = run_profiler(local_path)

        sessions[session_id]["status"] = "planning"
        router_result = router_node({
            "data_profile": profile,
            "business_query": business_query,
        })

        planner_result = planner_node({
            "data_profile": profile,
            "business_query": business_query,
            "router_output": router_result["router_output"],
        })

        plan = planner_result["analysis_plan"]

        sessions[session_id].update({
            "status": "awaiting_approval",
            "data_profile": profile,
            "router_output": router_result["router_output"],
            "analysis_plan": plan,
            "plan_approved_by_human": False,
            "current_step_index": 0,
            "active_code_block": None,
            "execution_success": False,
            "runtime_stdout": None,
            "runtime_stderr": None,
            "structured_metrics_output": {},
            "statistical_assumptions_passed": False,
            "critic_feedback": None,
            "retry_count": 0,
            "max_retries_per_step": 3,
            "error_history": [],
            "artifact_paths": [],
            "confidence_signals": {},
            "final_markdown_report": None,
            "sandbox_data_path": sandbox_path,
        })

    except Exception as e:
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)


def run_analysis_sync(session_id: str):
    from src.graph.nodes.coder import coder_node
    from src.graph.nodes.sandbox_node import sandbox_node
    from src.graph.nodes.critic import critic_node
    from src.graph.nodes.reporter import reporter_node

    session = sessions[session_id]
    plan = session["analysis_plan"]
    max_llm_calls = 20
    llm_calls = 0

    state = {
        "business_query": session["business_query"],
        "raw_data_path": session["sandbox_data_path"],
        "clean_data_path": None,
        "data_profile": session["data_profile"],
        "router_output": session["router_output"],
        "analysis_plan": plan,
        "plan_approved_by_human": True,
        "current_step_index": session["current_step_index"],
        "active_code_block": None,
        "execution_success": False,
        "runtime_stdout": None,
        "runtime_stderr": None,
        "structured_metrics_output": {},
        "statistical_assumptions_passed": False,
        "critic_feedback": None,
        "retry_count": 0,
        "max_retries_per_step": session["max_retries_per_step"],
        "error_history": [],
        "artifact_paths": [],
        "confidence_signals": {},
        "final_markdown_report": None,
    }

    sessions[session_id]["status"] = "running"

    try:
        while state["current_step_index"] < len(plan):
            if llm_calls >= max_llm_calls:
                sessions[session_id]["status"] = "error"
                sessions[session_id]["error"] = "Safety limit: too many LLM calls"
                return

            current_idx = state["current_step_index"]

            coder_result = coder_node(state)
            state.update(coder_result)
            llm_calls += 1

            sandbox_result = sandbox_node(state)
            state.update(sandbox_result)

            critic_result = critic_node(state)
            state.update(critic_result)

            if state["retry_count"] >= state["max_retries_per_step"]:
                state["retry_count"] = 0
                state["current_step_index"] = current_idx + 1
                continue

            if not state["execution_success"] or not state["statistical_assumptions_passed"]:
                continue

        sessions[session_id]["status"] = "reporting"
        reporter_result = reporter_node(state)
        state.update(reporter_result)

        sessions[session_id].update({
            "status": "complete",
            "final_markdown_report": state["final_markdown_report"],
            "confidence_signals": state["confidence_signals"],
            "artifact_paths": state["artifact_paths"],
            "structured_metrics_output": state["structured_metrics_output"],
        })

    except Exception as e:
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)


# --- Request/Response models ---

class AnalysisRequest(BaseModel):
    business_query: str


class ApproveRequest(BaseModel):
    session_id: str


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "service": "AutoInsight API"}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    session_id = str(uuid.uuid4())
    local_path = UPLOAD_DIR / f"{session_id}_{file.filename}"

    with open(local_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    sandbox_path = f"/workspace/{file.filename}"
    try:
        copy_file_to_sandbox(str(local_path), sandbox_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    sessions[session_id] = {
        "session_id": session_id,
        "status": "uploaded",
        "filename": file.filename,
        "local_path": str(local_path),
        "sandbox_path": sandbox_path,
    }

    return {"session_id": session_id, "filename": file.filename, "status": "uploaded"}


@app.post("/analyze")
async def analyze(request: AnalysisRequest, background_tasks: BackgroundTasks, session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a CSV first.")

    session = sessions[session_id]
    if session["status"] != "uploaded":
        raise HTTPException(status_code=400, detail=f"Session is in state '{session['status']}'. Cannot start.")

    sessions[session_id]["business_query"] = request.business_query
    background_tasks.add_task(run_graph_sync, session_id)

    return {"session_id": session_id, "status": "started"}


@app.get("/status/{session_id}")
def get_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    response = {
        "session_id": session_id,
        "status": session.get("status"),
        "filename": session.get("filename"),
    }

    if session.get("status") == "awaiting_approval":
        response["analysis_plan"] = session.get("analysis_plan", [])
        response["data_profile"] = {
            "row_count": session["data_profile"]["row_count"],
            "column_count": session["data_profile"]["column_count"],
            "columns": session["data_profile"]["columns"],
        }

    if session.get("status") == "complete":
        response["confidence_signals"] = session.get("confidence_signals", {})

    if session.get("status") == "error":
        response["error"] = session.get("error")

    return response


@app.post("/approve")
async def approve_plan(request: ApproveRequest, background_tasks: BackgroundTasks):
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    if sessions[session_id]["status"] != "awaiting_approval":
        raise HTTPException(status_code=400, detail="No plan awaiting approval.")

    sessions[session_id]["plan_approved_by_human"] = True
    background_tasks.add_task(run_analysis_sync, session_id)

    return {"session_id": session_id, "status": "approved"}


@app.get("/report/{session_id}")
def get_report(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = sessions[session_id]
    if session.get("status") != "complete":
        raise HTTPException(status_code=400, detail=f"Report not ready. Status: {session.get('status')}")

    return {
        "session_id": session_id,
        "report": session.get("final_markdown_report"),
        "confidence_signals": session.get("confidence_signals", {}),
        "artifact_paths": session.get("artifact_paths", []),
    }


@app.get("/plan/{session_id}")
def get_plan(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "analysis_plan": sessions[session_id].get("analysis_plan", []),
        "status": sessions[session_id].get("status"),
    }