import json

connection_info = {
    "transport": "tcp",
    "ip": "0.0.0.0",
    "shell_port": 8888,
    "iopub_port": 8889,
    "stdin_port": 8890,
    "control_port": 8891,
    "hb_port": 8892,
    "signature_scheme": "hmac-sha256",
    "key": ""
}

connection_file = "/workspace/kernel.json"
with open(connection_file, "w") as f:
    json.dump(connection_info, f)

from ipykernel.kernelapp import IPKernelApp
IPKernelApp.launch_instance(argv=["-f", connection_file])