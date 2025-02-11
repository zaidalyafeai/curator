import json
import os
import tempfile
import zipfile

from datasets import Dataset

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

        # first unzip it using zipfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # first write it to a file
            with open(os.path.join(temp_dir, "output.zip"), "wb") as f:
                f.write(execution_output.files)

            breakpoint()
            # then unzip it
            with zipfile.ZipFile(os.path.join(temp_dir, "output.zip"), "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # then get the file
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

    # push to hub
    dataset.push_to_hub("pimpalgaonkar/chart-generation", private=True)
