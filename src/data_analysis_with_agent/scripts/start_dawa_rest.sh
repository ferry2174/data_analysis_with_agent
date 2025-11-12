#!/bin/bash

export APP_ENV=$1

APP_HOME=~/Program/data_analysis_with_agent

export PROMETHEUS_MULTIPROC_DIR=$APP_HOME/metrics_dir

mkdir -p $PROMETHEUS_MULTIPROC_DIR
rm -rf $PROMETHEUS_MULTIPROC_DIR/*.db

exec gunicorn -c $(python -c "import data_analysis_with_agent; print(data_analysis_with_agent.GUNICORN_CONF_PATH)") data_analysis_with_agent.backend.main:app
