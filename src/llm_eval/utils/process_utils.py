import torch
import requests
import subprocess
import time


def launch_server(model_name: str, args: dict) -> subprocess.Popen:
    """Launch a model server process with the given arguments. Args: model_name: Name or path of the model. args: Dictionary of server arguments. Returns: Popen object for the launched server process."""
    gpus = torch.cuda.device_count()
    command = (
        f"python -m sglang_router.launch_server"
        f" --model-path {model_name}"
        f" --chat-template {args['chat_template']}"
        f" --dp-size {min(gpus, args['dp_size'])}"
        f" --tp-size {args['tp_size']}"
        f" --mem-fraction-static 0.9"
        f" --router-policy round_robin"
        f" --max-running-requests 100"
        f" --port {args.get('port', 8000)}"
    )
    stdout_log = open("logs/server_stdout.log", "w")
    stderr_log = open("logs/server_stderr.log", "w")
    return subprocess.Popen(command.split(), stdout=stdout_log, stderr=stderr_log)

def wait_for_server(port: int, timeout: int) -> None:
    """Wait for a server to become ready on the specified port, up to a timeout. Args: port: Port number to check. timeout: Maximum time to wait in seconds. Raises: TimeoutError if the server is not ready in time."""
    base_url = f"http://localhost:{port}"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/v1/models", headers={"Authorization": "Bearer None"})
            if response.status_code == 200:
                print("Server is ready.")
                time.sleep(5)
                return
        except requests.RequestException:
            time.sleep(1)
    raise TimeoutError("Server did not become ready in time.")
