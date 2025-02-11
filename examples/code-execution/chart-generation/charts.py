import ast
import json
import os
import tarfile
import tempfile
from io import BytesIO

from datasets import Dataset, Image

from bespokelabs import curator


class ChartCodeExecutor(curator.CodeExecutor):
    """Chart Generation Code Executor."""

    def code(self, row):
        """Extract code string from a dataset row."""
        return """
import matplotlib.pyplot as plt
import json

# Read input data
data = json.loads(input())

# Create the chart
plt.figure(figsize=(10, 6))
plt.plot(data['x'], data['y'])
plt.title(data['title'])
plt.xlabel(data['xlabel'])
plt.ylabel(data['ylabel'])

# Save the plot
plt.savefig('chart.png')
plt.close()
"""

    def code_input(self, row):
        """Prepare chart data as input."""
        chart_data = {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10], "title": "Sample Line Chart", "xlabel": "X Axis", "ylabel": "Y Axis"}
        return json.dumps(chart_data)

    def code_output(self, row, execution_output):
        """Store execution results."""
        row["chart_generated"] = execution_output.files is not None

        # Create a file-like object from bytes
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert the escaped bytes string back to actual bytes
            if isinstance(execution_output.files, bytes):
                # Convert bytes to string, then evaluate as literal to get proper bytes
                files_str = execution_output.files.decode("utf-8")
                actual_bytes = ast.literal_eval(files_str)
                tar_bytes = BytesIO(actual_bytes)
            else:
                tar_bytes = BytesIO(execution_output.files)

            with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
                tar.extractall(path=temp_dir)
            with open(os.path.join(temp_dir, "chart.png"), "rb") as f:
                row["chart_image"] = f.read()

        return row


if __name__ == "__main__":
    # Initialize executor with multiprocessing backend
    executor = ChartCodeExecutor(backend="multiprocessing")

    # Create sample dataset
    dataset = Dataset.from_list([{"id": 1}])

    # Execute chart generation
    results = executor(dataset)

    # Cast the chart_image column to Image type
    results = results.cast_column("chart_image", Image())

    # Push to hub
    results.push_to_hub("pimpalgaonkar/chart-generation", private=True)
