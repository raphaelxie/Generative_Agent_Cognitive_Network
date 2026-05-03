#!/usr/bin/env bash
# Watches the broker arm to completion, then auto-starts the hub arm,
# then runs all offline analysis. Designed to run unattended in the
# background once the broker reverie.py session has been launched and
# Chrome has been reloaded onto the broker branch.
#
# The only manual action required from the user during this script's
# lifetime is a SECOND Chrome reload of http://localhost:8000/simulator_home
# after the broker survey lands and this script flips curr_sim_code.json
# to the hub branch.

set -u

BACKEND_DIR="/Users/mac/Documents/PythonNote/SocResearch/generative_agents-main/reverie/backend_server"
STORAGE_DIR="/Users/mac/Documents/PythonNote/SocResearch/generative_agents-main/environment/frontend_server/storage"
LOG_FILE="${BACKEND_DIR}/broker_probe_watcher.log"

BROKER_BRANCH="prepost_n15_calibration-1_post_broker_t2400"
HUB_BRANCH="prepost_n15_calibration-1_post_hub_t2400"
BASELINE_BRANCH="prepost_n15_calibration-1"

BROKER_SURVEY_CSV="${STORAGE_DIR}/${BROKER_BRANCH}/survey/perception_survey_t2400.csv"
HUB_SURVEY_CSV="${STORAGE_DIR}/${HUB_BRANCH}/survey/perception_survey_t2400.csv"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "${LOG_FILE}"
}

cd "${BACKEND_DIR}" || exit 1

log "watcher started"
log "waiting for broker survey at ${BROKER_SURVEY_CSV}"

while [ ! -f "${BROKER_SURVEY_CSV}" ]; do
  sleep 60
done

log "broker survey detected, sleeping 30s for save+fin to flush"
sleep 30

if [ -d "${STORAGE_DIR}/${HUB_BRANCH}" ]; then
  log "hub branch folder already exists; aborting hub launch to avoid copy-fail"
else
  log "launching hub arm reverie.py with hub_commands.txt"
  nohup python reverie.py < hub_commands.txt > "${BACKEND_DIR}/hub_reverie.log" 2>&1 &
  HUB_PID=$!
  log "hub reverie pid=${HUB_PID}"
  log "ACTION REQUIRED: reload http://localhost:8000/simulator_home in Chrome to begin hub branch stepping"
  osascript -e 'display notification "Reload http://localhost:8000/simulator_home in Chrome to start the hub-isolation arm." with title "Broker Probe Watcher" sound name "Glass"' >/dev/null 2>&1 || true
fi

log "waiting for hub survey at ${HUB_SURVEY_CSV}"
while [ ! -f "${HUB_SURVEY_CSV}" ]; do
  sleep 60
done
log "hub survey detected, sleeping 30s for save+fin to flush"
sleep 30

log "running offline analysis: broker analyze_survey"
python analyze_survey.py "${STORAGE_DIR}/${BROKER_BRANCH}/survey" >> "${LOG_FILE}" 2>&1

log "running offline analysis: broker network_summary"
python survey_network_summary.py "${STORAGE_DIR}/${BROKER_BRANCH}/survey" >> "${LOG_FILE}" 2>&1

log "running offline analysis: hub analyze_survey"
python analyze_survey.py "${STORAGE_DIR}/${HUB_BRANCH}/survey" >> "${LOG_FILE}" 2>&1

log "running offline analysis: hub network_summary"
python survey_network_summary.py "${STORAGE_DIR}/${HUB_BRANCH}/survey" >> "${LOG_FILE}" 2>&1

log "WATCHER COMPLETE -- pre/post broker probe data and analyses are ready"
log "  pre baseline: ${STORAGE_DIR}/${BASELINE_BRANCH}/survey/"
log "  post broker:  ${STORAGE_DIR}/${BROKER_BRANCH}/survey/"
log "  post hub:     ${STORAGE_DIR}/${HUB_BRANCH}/survey/"
osascript -e 'display notification "Pre/post broker probe complete. All surveys and analyses are written." with title "Broker Probe Watcher" sound name "Glass"' >/dev/null 2>&1 || true
