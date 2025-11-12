.PHONY: contrib quality style help html

check_dirs := src utils setup.py

package ?= ""

quality:
	-ruff check $(check_dirs)  # linter
	-ruff format --check $(check_dirs)   # formatter
	ruff format $(check_dirs)   # formatter
	python utils/check_static_imports.py --package $(package)
#	mypy src  # type checker

style:
	python utils/check_static_imports.py --update --package $(package)
	ruff format --check $(check_dirs)   # formatter

clean:
	rm -rf build/* dist/* data_analysis_with_agent.egg-info/* src/data_analysis_with_agent.egg-info/*

package:
	python setup.py sdist bdist_wheel

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXPYAPIDOC ?= sphinx-apidoc
SOURCEDIR     = docs
BUILDDIR      = docs/_build
PYSOURCEDIR   = src/data_analysis_with_agent

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
#apidoc:
#	@$(SPHINXPYAPIDOC) -o source "$(PYSOURCEDIR)" 

html: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# 定义路径
ROOT_DIR = src/data_analysis_with_agent/backend
FRONTEND_DIR = src/data_analysis_with_agent/frontend
DIST_DIR = dist/data_analysis_with_agent
ASSETS_DIR = $(FRONTEND_DIR)/assets
TEMPLATES_DIR = $(FRONTEND_DIR)/templates
STATIC_DIR = $(FRONTEND_DIR)/static
PYTHON_FILES = $(shell find $(ROOT_DIR) -name "*.py")
HTML_FILES = $(shell find $(TEMPLATES_DIR) -name "*.html")
JS_FILES = $(shell find $(STATIC_DIR) -name "*.js")
CSS_FILES = $(shell find $(STATIC_DIR) -name "*.css")

# 构建目标
data_analysis_with_agent: compile_py obfuscate_html obfuscate_js obfuscate_css copy_assets

# 编译 Python 文件为 .pyc
compile_py:
	@echo "Compiling Python files..."
	@mkdir -p $(DIST_DIR)
	@find $(PYTHON_FILES) | while read file; do \
		dest_file=$$(echo $$file | sed 's|$(ROOT_DIR)|$(DIST_DIR)|' | sed 's|\.py$$|.pyc|'); \
		dest_dir=$$(dirname $$dest_file); \
		mkdir -p $$dest_dir; \
		python -m py_compile $$file; \
		compiled_file=$$(dirname $$file)/__pycache__/$$(basename $$file .py).cpython-311.pyc; \
		mv $$compiled_file $$dest_file; \
	done

# 混淆 HTML 文件
obfuscate_html:
	@echo "Obfuscating HTML files..."
	@for file in $(HTML_FILES); do \
		dest_file=$$(echo $$file | sed 's|$(ROOT_DIR)|$(DIST_DIR)|'); \
		dest_dir=$$(dirname $$dest_file); \
		mkdir -p $$dest_dir; \
		npx html-minifier --collapse-whitespace --remove-comments --remove-script-type-attributes --remove-style-link-type-attributes --minify-css true --minify-js true $$file -o $$dest_file; \
	done

# 混淆 JavaScript 文件
obfuscate_js:
	@echo "Obfuscating JavaScript files..."
	@for file in $(JS_FILES); do \
		dest_file=$$(echo $$file | sed 's|$(ROOT_DIR)|$(DIST_DIR)|'); \
		dest_dir=$$(dirname $$dest_file); \
		mkdir -p $$dest_dir; \
		npx uglifyjs $$file -o $$dest_file; \
	done

# 混淆 CSS 文件
obfuscate_css:
	@echo "Obfuscating CSS files..."
	@for file in $(CSS_FILES); do \
		dest_file=$$(echo $$file | sed 's|$(ROOT_DIR)|$(DIST_DIR)|'); \
		dest_dir=$$(dirname $$dest_file); \
		mkdir -p $$dest_dir; \
		npx clean-css-cli -o $$dest_file $$file; \
	done

# 复制 assets 文件夹内容
copy_assets:
	@echo "Copying assets..."
	@mkdir -p $(DIST_DIR)/assets
	@cp -r $(ASSETS_DIR)/* $(DIST_DIR)/assets/

# 清理生成的文件
data_analysis_with_agent_clean:
	@echo "Cleaning up..."
	@rm -rf $(DIST_DIR)

.PHONY: data_analysis_with_agent compile_py obfuscate_html obfuscate_js obfuscate_css copy_assets data_analysis_with_agent_clean
