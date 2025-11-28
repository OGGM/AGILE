#!/usr/bin/env bash
#
# Run an AGILE/OGGM idealized experiment locally using Docker.
#
# What this script does:
#  - Creates a local working directory (HOST_WORKDIR) if needed.
#  - Clones the AGILE repository there (if not already present).
#  - Pulls the AGILE Docker image.
#  - Starts a container, sets up a Python virtual environment,
#    installs a specific AGILE commit, and runs:
#       run_idealized_experiment --params_file ... --experiment_file ...
#
# Usage:
#   chmod +x run_example_experiment.sh
#   ./run_example_experiment.sh
#
# Optional environment variables:
#   HOST_WORKDIR  - Directory on the host used for code, data and outputs.
#                   Default: ./agile_workdir (relative to where you run the script).
#   AGILE_IMAGE   - Docker image tag to use.
#                   Default: ghcr.io/oggm/agile:20230525
#
# To change which experiment is run, adjust RUN_SCRIPTS_SUBDIR,
# PARAMS_FILE and EXPERIMENT_FILE below.

set -euo pipefail

# --- Configuration -----------------------------------------------------------

# Local directory where the AGILE repo, inputs and outputs will live
HOST_WORKDIR="${HOST_WORKDIR:-$PWD/agile_workdir}"

# Docker image to use
AGILE_IMAGE="${AGILE_IMAGE:-ghcr.io/oggm/agile:20230525}"

# Repository from which the Python package AGILE will be installed
AGILE_REPO_URL="https://github.com/OGGM/AGILE.git"

# Git commit of AGILE to install inside the container (for reproducibility)
AGILE_COMMIT="330903def7bb612f495ec58db79097c07bfd0613"

# Subdirectory (inside HOST_WORKDIR) that contains the experiment script & params
RUN_SCRIPTS_SUBDIR="AGILE/agile1d/sandbox/paper_v01_code/minimal_run_example"

# Filenames used by the experiment run
PARAMS_FILE="params.cfg"
EXPERIMENT_FILE="mini_experiment_file_fg_oggm.py"

# --- Basic checks ------------------------------------------------------------

# Require docker on the host
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: 'docker' command not found. Please install Docker and try again." >&2
  exit 1
fi

# Require git on the host (to clone AGILE)
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: 'git' command not found. Please install git and try again." >&2
  exit 1
fi

# --- Prepare host working directory ------------------------------------------

echo "Using host working directory: ${HOST_WORKDIR}"
mkdir -p "${HOST_WORKDIR}"
cd "${HOST_WORKDIR}"

# Clone AGILE repo if it isn't there yet
if [ ! -d "AGILE" ]; then
  echo "Cloning AGILE repository into ${HOST_WORKDIR}/AGILE ..."
  git clone "${AGILE_REPO_URL}" AGILE
else
  echo "AGILE repository already exists in ${HOST_WORKDIR}/AGILE – not cloning."
fi

# Check that the run scripts directory exists
if [ ! -d "${RUN_SCRIPTS_SUBDIR}" ]; then
  echo "ERROR: run scripts directory not found at:"
  echo "  ${HOST_WORKDIR}/${RUN_SCRIPTS_SUBDIR}"
  echo "Make sure the AGILE repo structure matches the expected layout."
  exit 1
fi

# Check that the experiment & params files exist
if [ ! -f "${RUN_SCRIPTS_SUBDIR}/${EXPERIMENT_FILE}" ]; then
  echo "ERROR: experiment file not found:"
  echo "  ${HOST_WORKDIR}/${RUN_SCRIPTS_SUBDIR}/${EXPERIMENT_FILE}"
  exit 1
fi

if [ ! -f "${RUN_SCRIPTS_SUBDIR}/${PARAMS_FILE}" ]; then
  echo "ERROR: params file not found:"
  echo "  ${HOST_WORKDIR}/${RUN_SCRIPTS_SUBDIR}/${PARAMS_FILE}"
  exit 1
fi

# --- Pull Docker image -------------------------------------------------------

echo "Pulling AGILE Docker image: ${AGILE_IMAGE}"
docker pull "${AGILE_IMAGE}"

# --- Run container & experiment ----------------------------------------------

echo "Starting AGILE container and running idealized experiment..."
echo "This may take a while the first time as some data and Python packages need to be downloaded."

docker run --rm -i \
  --user "$(id -u):$(id -g)" \
  -e OGGM_WORKDIR=/work \
  -e EXPERIMENT_FILE="/work/${RUN_SCRIPTS_SUBDIR}/${EXPERIMENT_FILE}" \
  -e PARAMS_FILE="/work/${RUN_SCRIPTS_SUBDIR}/${PARAMS_FILE}" \
  -e AGILE_COMMIT="${AGILE_COMMIT}" \
  --ulimit nofile=65000:65000 \
  -v "${HOST_WORKDIR}":/work \
  "${AGILE_IMAGE}" \
  bash -s <<'EOF'
set -e

# Inside the container, OGGM_WORKDIR is /work (set above)

# Use an isolated home directory inside the mounted work folder,
# so user-specific files stay under /work and are created with your UID.
export HOME="${OGGM_WORKDIR}/fake_home"
mkdir -p "${HOME}"

# Directory where run_idealized_experiment will write its outputs
export OUTPUT_DIR="${OGGM_WORKDIR}/experiment_results"
mkdir -p "${OUTPUT_DIR}"

# Remove any existing virtual environment to start from a clean state
rm -rf "${OGGM_WORKDIR}/oggm_env"

# Create a virtual environment which also sees system site-packages
python3 -m venv --system-site-packages "${OGGM_WORKDIR}/oggm_env"

# Some venvs only provide "python", but AGILE expects a "python3" binary
if [ ! -x "${OGGM_WORKDIR}/oggm_env/bin/python3" ] && [ -x "${OGGM_WORKDIR}/oggm_env/bin/python" ]; then
  ln -s python "${OGGM_WORKDIR}/oggm_env/bin/python3"
fi

# Activate the virtual environment
# shellcheck source=/dev/null
source "${OGGM_WORKDIR}/oggm_env/bin/activate"

# Upgrade pip and setuptools inside the virtual environment
pip install --upgrade pip setuptools

# Install AGILE from the chosen Git commit for reproducible runs
pip install --no-deps --force-reinstall \
  "git+https://github.com/OGGM/AGILE.git@${AGILE_COMMIT}"

# Increase the maximum number of open file descriptors (best effort)
ulimit -n 65000 || true

# Change into the directory that contains the experiment script
cd "$(dirname "$EXPERIMENT_FILE")"

# Run the idealized experiment
run_idealized_experiment \
  --params_file "$PARAMS_FILE" \
  --experiment_file "$EXPERIMENT_FILE" \
  --first_guess oggm \
  --output_folder "${OUTPUT_DIR}"
EOF

echo
echo "Experiment finished."
echo "Inputs, outputs and logs are in: ${HOST_WORKDIR}"

