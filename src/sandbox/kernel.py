import time
import json
import os
from jupyter_client import BlockingKernelClient


SANDBOX_HOST = os.getenv("SANDBOX_HOST", "localhost")
SANDBOX_PORT = int(os.getenv("SANDBOX_PORT", "8888"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "60"))

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


def execute_code(code: str) -> dict:
    client = get_kernel_client()

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

    return {
        "execution_success": execution_success,
        "stdout": stdout,
        "stderr": stderr,
        "metrics": metrics,
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