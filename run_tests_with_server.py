import subprocess
import time
import sys
import os
import requests
import signal

def wait_for_server(url, retries=30, delay=2):
    print(f"Waiting for server at {url}...")
    for i in range(retries):
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(delay)
    print("Server failed to start.")
    return False

def main():
    # Start the server
    print("Starting SageDB server...")
    # Use the current python executable to ensure we use the venv
    python_exe = sys.executable
    server_process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "main:app", "--port", "8000"],
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Wait for server to start
        if not wait_for_server("http://localhost:8000"):
            print("Server failed to start. Server output:")
            try:
                outs, errs = server_process.communicate(timeout=5)
                print(outs)
                print(errs)
            except:
                pass
            server_process.kill()
            sys.exit(1)

        # Run tests
        print("\nRunning tests...")
        test_process = subprocess.run(
            [python_exe, "test_sagedb.py"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True
        )
        
        print(test_process.stdout)
        if test_process.stderr:
            print("Test Errors:")
            print(test_process.stderr)

        if test_process.returncode != 0:
            print("Tests failed!")
            sys.exit(1)
        else:
            print("Tests passed!")

    finally:
        print("Stopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()

if __name__ == "__main__":
    main()
