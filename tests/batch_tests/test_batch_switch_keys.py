import pytest
import time
import os
from tests.batch_tests.test_helpers import run_script
from tests.batch_tests.test_helpers import prepare_test_cache

"""
USAGE:
pytest -s tests/batch_tests/test_batch_switch_keys.py
"""


@pytest.mark.cache_dir(os.path.expanduser("~/.cache/curator-tests/test-batch-switch-keys"))
@pytest.mark.usefixtures("prepare_test_cache")
def test_batch_resume():
    script = [
        "python",
        "tests/batch_tests/simple_batch.py",
        "--log-level",
        "DEBUG",
        "--n-requests",
        "2",
        "--batch-size",
        "1",
        "--batch-check-interval",
        "10",
    ]

    env = os.environ.copy()

    # First run should process 1 batch and exit
    print("FIRST RUN")

    env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_1"]
    stop_line_pattern = r"Marked batch ID batch_[a-f0-9]{32} as downloaded"
    output1, _ = run_script(script, stop_line_pattern, env=env)
    print(output1)

    # Small delay to ensure files are written
    time.sleep(1)

    # Second run should process the remaining batch
    print("SECOND RUN")
    env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_2"]
    output2, _ = run_script(script, env=env)
    print(output2)

    # checks
    assert "1 out of 1 batches already downloaded." in output2
