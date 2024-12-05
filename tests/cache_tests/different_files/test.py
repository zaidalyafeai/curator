import pytest
import subprocess
import shutil


def run_script(script_path, delete_cache=False):
    command = ["python", script_path]
    if delete_cache:
        command.append("--delete-cache")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    lines = ""
    for line in process.stderr:
        lines += line

    for line in process.stdout:
        lines += line

    process.wait()
    return lines


@pytest.fixture
def clean_caches():
    """Fixture to ensure clean caches before tests"""
    # Delete caches before test --> provide a differerent CURATOR_CACHE_DIR
    shutil.rmtree("~/.cache/curator", ignore_errors=True)

    # Run test
    yield

    # Clean up after test (optional, but good practice)
    shutil.rmtree("~/.cache/curator", ignore_errors=True)


def test_cache_behavior(clean_caches):
    cache_hit_log = "Using cached output dataset."

    # Run one.py twice and check for cache behavior
    print("RUNNING ONE.PY")
    output_one_first = run_script("tests/cache_tests/different_files/one.py")
    print(output_one_first)
    assert cache_hit_log not in output_one_first, "First run of one.py should not hit cache"
    print("RUNNING ONE.PY AGAIN")
    output_one_second = run_script("tests/cache_tests/different_files/one.py")
    print(output_one_second)
    assert cache_hit_log in output_one_second, "Second run of one.py should hit cache"

    # Run two.py and check for cache behavior
    print("RUNNING TWO.PY")
    output_two = run_script("tests/cache_tests/different_files/two.py")
    print(output_two)
    assert cache_hit_log in output_two, "First run of two.py should hit cache"
