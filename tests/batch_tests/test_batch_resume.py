import pytest
import subprocess
import time
import shutil
import os
import re


def run_script(script, stop_line_pattern=None):
    process = subprocess.Popen(script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    lines = ""
    for line in process.stderr:
        print(line, end="")  # Print each line as it is received
        lines += line
        if stop_line_pattern and re.search(stop_line_pattern, line):
            process.terminate()
            break

    for line in process.stdout:
        print(line, end="")  # Print each line as it is received
        lines += line
        if stop_line_pattern and re.search(stop_line_pattern, line):
            process.terminate()
            break

    process.wait()
    return lines, process.returncode


@pytest.fixture
def clean_caches():
    """Fixture to ensure clean caches before tests"""
    cache_dir = os.path.expanduser("~/.cache/curator")
    # Delete caches before test
    shutil.rmtree(cache_dir, ignore_errors=True)

    # Run test
    yield

    # Clean up after test
    shutil.rmtree(cache_dir, ignore_errors=True)


def test_batch_resume(clean_caches):
    # First run should process 1 batch and exit
    print("FIRST RUN")
    stop_line_pattern = r"Marked batch ID batch_[a-f0-9]{32} as downloaded"
    output1, return_code1 = run_script(
        ["python", "tests/batch_tests/three_small_batches.py", "--log-level", "DEBUG"],
        stop_line_pattern,
    )
    print(output1)

    # Small delay to ensure files are written
    time.sleep(1)

    # Second run should process the remaining batch
    print("SECOND RUN")
    output2, return_code1 = run_script(
        ["python", "tests/batch_tests/three_small_batches.py", "--log-level", "DEBUG"]
    )
    print(output2)
    assert "2 out of 2 remaining batches are already submitted." in output2
    assert "1 out of 1 batches already downloaded." in output2


# pytest -s tests/batch_tests/test_batch_resume.py
