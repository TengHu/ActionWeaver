import pdb
import subprocess
from typing import List

"""This script is utilized to verify the proper functioning of the notebooks listed below. 
It relies on nbmake; refer to https://github.com/treebeardtech/nbmake?tab=readme-ov-file for more information.
If a cell is tagged with 'skip-execution', it will not be executed. Utilize the property inspector in Jupyter Lab to add tags to cells.

TODO: add checks for cell outputs.

"""

NOTEBOOKS = [
    "docs/source/notebooks//cookbooks/cookbook.ipynb",
    "docs/source/notebooks//cookbooks/quickstart.ipynb",
    "docs/source/notebooks//cookbooks/extract_tabular_data.ipynb",
    "docs/source/notebooks//cookbooks/pydantic.ipynb",
    "docs/source/notebooks//cookbooks/orchestration.ipynb",
    "docs/source/notebooks//cookbooks/stateful_agent.ipynb",
]


ALL_USER_VISIBLE_NOTEBOOKS = NOTEBOOKS + [
    # "docs/source/notebooks//cookbooks/langsmith.ipynb",
    # "docs/source/notebooks//cookbooks/parallel_tools.ipynb",
    # "docs/source/notebooks//cookbooks/knowledge_graph_extraction.ipynb",
    # "docs/source/notebooks//cookbooks/logging.ipynb",
    # "docs/source/notebooks//cookbooks/wrapper.ipynb",
]


def find_ipynb_files(directory: str):
    try:
        result = subprocess.run(
            ["find", directory, "-type", "f", "-name", "*.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        ipynb_files = result.stdout.splitlines()
        return ipynb_files
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return []


def execute_notebooks(notebook_path: List[str]):

    # allows each cell 30 seconds to finis
    command = f"pytest --nbmake --nbmake-timeout=30 {' '.join(notebook_path)}"

    # Execute the command
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    directory = "docs/source/notebooks/"
    ipynb_files = find_ipynb_files(directory)

    substrings_to_exclude = ["ipynb_checkpoints"]
    ipynb_files = [
        file
        for file in ipynb_files
        if not any(substring in file for substring in substrings_to_exclude)
    ]

    extra_files = [file for file in ipynb_files if file not in NOTEBOOKS]
    if extra_files:
        print("\nExtra files found:")
        for file in extra_files:
            print(file)
        decision = input("Do you want to continue? (yes/no): ")
        if decision.lower() != "yes":
            print("Exiting.")
            exit(0)

    execute_notebooks(NOTEBOOKS)
