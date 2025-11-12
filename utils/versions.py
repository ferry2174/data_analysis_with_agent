import os


def get_version() -> str:
    abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src/data_analysis_with_agent/__init__.py")
    with open(abs_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")
