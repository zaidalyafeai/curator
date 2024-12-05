import pytest
import subprocess
import time


def run_script(script, stop_line=None):
    process = subprocess.Popen(
        script,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    lines = ""
    for line in process.stderr:
        print(line, end='')  # Print each line as it is received
        lines += line
        if stop_line and stop_line in line:
            process.terminate()
            break

    for line in process.stdout:
        print(line, end='')  # Print each line as it is received
        lines += line
        if stop_line and stop_line in line:
            process.terminate()
            break

    process.wait()
    return lines, process.returncode


# @pytest.fixture
# def clean_caches():
#     """Fixture to ensure clean caches before tests"""
#     # Delete caches before test
#     run_script(["python", "tests/batch_tests/three_small_batches.py", "--delete-cache", "--log-level", "DEBUG"])

#     # Run test
#     yield

#     # Clean up after test (optional, but good practice)
#     run_script(["python", "tests/batch_tests/three_small_batches.py", "--delete-cache", "--log-level", "DEBUG"])


def test_batch_resume():
    # First run should process 2 batches and exit
    print("FIRST RUN")
    stop_line = "Batches returned: 1/3"
    output1, return_code1 = run_script(["python", "tests/batch_tests/three_small_batches.py", "--log-level", "DEBUG"], stop_line)
    print(output1)
    

    
    # Small delay to ensure files are written
    time.sleep(1)
    
    # Second run should process the remaining batch
    print("SECOND RUN")
    output2, return_code2 = run_script("tests/batch_tests/three_small_batches.py")
    print(output2)
    
    # Check that the last batch was processed
    assert "Processed batch 3" in output2
    assert "Processed batch 1" not in output2
    assert "Processed batch 2" not in output2 