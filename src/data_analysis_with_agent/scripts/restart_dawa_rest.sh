APP_HOME=~/Program/data_analysis_with_agent

export PROMETHEUS_MULTIPROC_DIR=$APP_HOME/metrics_dir

mkdir -p $PROMETHEUS_MULTIPROC_DIR
rm -rf $PROMETHEUS_MULTIPROC_DIR/*.db

kill -HUP $(cat ~/Program/data_analysis_with_agent/.app_pid)
