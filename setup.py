import os

from setuptools import find_packages, setup


def get_version() -> str:
    abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/data_analysis_with_agent/__init__.py")
    with open(abs_path, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

install_requires = [
    "fastapi==0.116.1",
    "uvicorn==0.35.0",
    "gunicorn==23.0.0",
    "prometheus-client==0.22.1",
    "numpy==1.26.4",
    "pandas==2.3.1",
    "requests==2.32.5",
    "pydantic==2.11.7",
    "concurrent-log-handler==0.9.28",
    "pyyaml==6.0.2",
    "redis==6.4.0",
    "prometheus-client==0.22.1",
    "aiomysql==0.2.0",
    "aiokafka==0.12.0",
    "langchain-community==0.4.1",
    "matplotlib==3.10.7",
    "plotly==6.4.0",
    "gradio==5.49.1",
    "python-louvain==0.16",
    "orjson==3.11.2",
    "langchain_deepseek==1.0.0",
    "langgraph==1.0.3",
]

extras = {}

# Typing extra dependencies list is duplicated in `.pre-commit-config.yaml`
# Please make sure to update the list there when adding a new typing dependency.
extras["typing"] = [
    "typing-extensions>=4.8.0",
    "types-PyYAML",
    "types-requests",
    "types-simplejson",
    "types-toml",
    "types-tqdm",
    "types-urllib3",
]

extras["quality"] = [
    "ruff>=0.5.0",
    "mypy==1.5.1",
    "libcst==1.4.0",
]

extras["doc"] = [
    "sphinx>=8.1.0",
    "sphinx-autobuild>=2024.10.3",
    "sphinx_rtd_theme>=3.0.0",
    "myst_parser>=4.0.0",
    "sphinx-markdown-tables>=0.0.17",
]

extras["dev"] = [
    "pysocks>=1.7.1",
]

extras["all"] = extras["quality"] + extras["typing"] + extras["doc"] + extras["dev"]

package_data = {
    "": [],
}

exclude_package_data = {
    "": [],
}

# 获取脚本路径
script_dir = os.path.join('src', 'data_analysis_with_agent', 'scripts')
scripts = [os.path.join(script_dir, f) for f in os.listdir(script_dir) if f.endswith('.sh')]

setup(
    name="data_analysis_with_agent",
    version=get_version(),
    author="ZhangJingbo",
    author_email="zhangjb@eulernet.com",
    description="Eulernet data features unified access service",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="communication video machine-learning models natural-language-processing deep-learning pytorch pretrained-models",
    license="",
    url="",
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
    ),
    include_package_data=True,
    package_data=package_data,
    exclude_package_data=exclude_package_data,
    extras_require=extras,
    scripts=scripts,
    python_requires=">=3.10.0",
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
