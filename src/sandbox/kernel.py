import time
import json
import os
import subprocess
from jupyter_client import BlockingKernelClient


SANDBOX_HOST = os.getenv("SANDBOX_HOST", "localhost")
SANDBOX_PORT = int(os.getenv("SANDBOX_PORT", "8888"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "60"))
SANDBOX_CONTAINER = os.getenv("SANDBOX_CONTAINER", "autoinsight-sandbox")
HOST_ARTIFACTS_DIR = os.getenv("HOST_ARTIFACTS_DIR", "workspace/artifacts")

_client: BlockingKernelClient | None = None


def get_kernel_client() -> BlockingKernelClient:
    global _client
    if _client is not None:
        return _client

    print(f"[Kernel] Connecting to sandbox at {SANDBOX_HOST}:{SANDBOX_PORT}...")

    connection_info = {
        "transport": "tcp",
        "ip": SANDBOX_HOST,
        "shell_port": SANDBOX_PORT,
        "iopub_port": SANDBOX_PORT + 1,
        "stdin_port": SANDBOX_PORT + 2,
        "control_port": SANDBOX_PORT + 3,
        "hb_port": SANDBOX_PORT + 4,
        "signature_scheme": "hmac-sha256",
        "key": "",
    }

    client = BlockingKernelClient()
    client.load_connection_info(connection_info)
    client.start_channels()

    try:
        client.wait_for_ready(timeout=30)
        print("[Kernel] Connected to IPython kernel.")
    except Exception as e:
        raise RuntimeError(f"Could not connect to kernel: {e}")

    _client = client
    return client


def reset_kernel_client():
    global _client
    if _client is not None:
        try:
            _client.stop_channels()
        except Exception:
            pass
        _client = None


def _list_sandbox_artifacts(client: BlockingKernelClient) -> list[str]:
    """List filenames currently in /workspace/artifacts inside the sandbox kernel."""
    listing = execute_code_raw(
        client,
        """
import os, json
path = '/workspace/artifacts'
files = sorted(os.listdir(path)) if os.path.isdir(path) else []
print(json.dumps(files))
"""
    )
    try:
        return json.loads(listing.strip())
    except (json.JSONDecodeError, ValueError):
        return []


def _copy_new_artifacts_to_host(new_filenames: list[str]) -> list[str]:
    """docker cp each newly-created artifact file from the sandbox container to the host."""
    os.makedirs(HOST_ARTIFACTS_DIR, exist_ok=True)
    host_paths = []
    for filename in new_filenames:
        container_path = f"{SANDBOX_CONTAINER}:/workspace/artifacts/{filename}"
        host_path = os.path.join(HOST_ARTIFACTS_DIR, filename)
        try:
            subprocess.run(
                ["docker", "cp", container_path, host_path],
                check=True,
                capture_output=True,
                timeout=30,
            )
            host_paths.append(host_path)
            print(f"[Kernel] Retrieved artifact: {filename}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[Kernel] Failed to retrieve artifact '{filename}': {e}")
    return host_paths


def execute_code(code: str) -> dict:
    client = get_kernel_client()

    artifacts_before = set(_list_sandbox_artifacts(client))

    stdout_lines = []
    stderr_lines = []
    execution_success = False

    try:
        msg_id = client.execute(code)
        deadline = time.time() + EXECUTION_TIMEOUT

        while time.time() < deadline:
            try:
                msg = client.get_iopub_msg(timeout=5)
                msg_type = msg["msg_type"]
                content = msg.get("content", {})

                if msg_type == "stream":
                    if content.get("name") == "stdout":
                        stdout_lines.append(content.get("text", ""))
                    elif content.get("name") == "stderr":
                        stderr_lines.append(content.get("text", ""))

                elif msg_type == "error":
                    traceback = content.get("traceback", [])
                    stderr_lines.extend(traceback)

                elif msg_type == "status":
                    if content.get("execution_state") == "idle":
                        execution_success = True
                        break

            except Exception:
                break

        if time.time() >= deadline:
            stderr_lines.append(f"Execution timed out after {EXECUTION_TIMEOUT} seconds")
            execution_success = False

    except Exception as e:
        stderr_lines.append(f"Kernel communication error: {e}")
        execution_success = False
        reset_kernel_client()

    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)

    if stderr and not stdout:
        execution_success = False

    metrics = {}
    metrics_path = "/workspace/metrics.json"
    try:
        read_result = execute_code_raw(
            client,
            f"""
import json, os
if os.path.exists('{metrics_path}'):
    with open('{metrics_path}') as f:
        print(json.dumps(json.load(f)))
else:
    print('{{}}')
"""
        )
        metrics_text = read_result.strip()
        if metrics_text:
            metrics = json.loads(metrics_text)
    except Exception:
        metrics = {}

    artifacts = []
    try:
        artifacts_after = set(_list_sandbox_artifacts(client))
        new_filenames = sorted(artifacts_after - artifacts_before)
        if new_filenames:
            artifacts = _copy_new_artifacts_to_host(new_filenames)
    except Exception as e:
        print(f"[Kernel] Artifact retrieval skipped due to error: {e}")

    return {
        "execution_success": execution_success,
        "stdout": stdout,
        "stderr": stderr,
        "metrics": metrics,
        "artifacts": artifacts,
    }


def execute_code_raw(client: BlockingKernelClient, code: str) -> str:
    client.execute(code)
    output_lines = []
    deadline = time.time() + 10

    while time.time() < deadline:
        try:
            msg = client.get_iopub_msg(timeout=2)
            if msg["msg_type"] == "stream" and msg["content"].get("name") == "stdout":
                output_lines.append(msg["content"].get("text", ""))
            elif msg["msg_type"] == "status" and msg["content"].get("execution_state") == "idle":
                break
        except Exception:
            break

    return "".join(output_lines)


def sandbox_health_check() -> bool:
    try:
        client = get_kernel_client()
        result = execute_code_raw(client, "print('health_ok')")
        return "health_ok" in result
    except Exception:
        return False
