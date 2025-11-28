#!/usr/bin/env bash
#
# Run a Python plotting script inside the AGILE Docker image, using the
# experiment outputs from a previous run and storing figures in a chosen folder.
#
# Usage:
#   chmod +x create_example_plot.sh
#   ./create_example_plot.sh
#
# Optional environment variables:
#   HOST_WORKDIR  - Directory with AGILE repo + experiment outputs
#                   Default: ./agile_workdir
#   AGILE_IMAGE   - Docker image to use
#                   Default: ghcr.io/oggm/agile:20230525
#   OUTPUT_DIR    - Directory containing experiment outputs
#                   Default: $HOST_WORKDIR/output
#   FIGURES_DIR   - Directory where figures should be written
#                   Default: $HOST_WORKDIR/figures
#
# The Python script can access OUTPUT_DIR and FIGURES_DIR via:
#   os.environ["OUTPUT_DIR"]
#   os.environ["FIGURES_DIR"]

set -euo pipefail

# --- Configuration -----------------------------------------------------------

HOST_WORKDIR="${HOST_WORKDIR:-$PWD/agile_workdir}"
AGILE_IMAGE="${AGILE_IMAGE:-ghcr.io/oggm/agile:20230525}"

# Where the plotting script lives inside HOST_WORKDIR
RUN_SCRIPTS_SUBDIR="AGILE/agile1d/sandbox/paper_v01_code/minimal_run_example"

# Plotting script name
PLOT_SCRIPT="plot_example_fig.py"

# Directories for input (experiment outputs) and output (figures)
OUTPUT_DIR="${OUTPUT_DIR:-${HOST_WORKDIR}/output}"
FIGURES_DIR="${FIGURES_DIR:-${HOST_WORKDIR}/figures}"

# --- Basic checks ------------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: 'docker' not found. Please install Docker."
  exit 1
fi

echo "HOST_WORKDIR  = ${HOST_WORKDIR}"
echo "OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "FIGURES_DIR   = ${FIGURES_DIR}"

# Ensure HOST_WORKDIR exists
if [ ! -d "${HOST_WORKDIR}" ]; then
  echo "ERROR: HOST_WORKDIR does not exist: ${HOST_WORKDIR}"
  echo "Run the experiment script first."
  exit 1
fi

cd "${HOST_WORKDIR}"

# Ensure AGILE repo exists
if [ ! -d "AGILE" ]; then
  echo "ERROR: AGILE repository missing at ${HOST_WORKDIR}/AGILE"
  exit 1
fi

# Ensure plotting script exists
if [ ! -f "${RUN_SCRIPTS_SUBDIR}/${PLOT_SCRIPT}" ]; then
  echo "ERROR: plotting script not found:"
  echo "  ${HOST_WORKDIR}/${RUN_SCRIPTS_SUBDIR}/${PLOT_SCRIPT}"
  exit 1
fi

# Ensure OUTPUT_DIR exists
if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "ERROR: OUTPUT_DIR does not exist: ${OUTPUT_DIR}"
  exit 1
fi

# Create FIGURES_DIR if needed
mkdir -p "${FIGURES_DIR}"

# Ensure both dirs are inside HOST_WORKDIR (required for /work mount)
case "${OUTPUT_DIR}" in
  ${HOST_WORKDIR}/*) ;;
  *) echo "ERROR: OUTPUT_DIR must be inside HOST_WORKDIR."; exit 1;;
esac

case "${FIGURES_DIR}" in
  ${HOST_WORKDIR}/*) ;;
  *) echo "ERROR: FIGURES_DIR must be inside HOST_WORKDIR."; exit 1;;
esac

# Compute container-visible paths
OUTPUT_DIR_IN_CONTAINER="/work${OUTPUT_DIR#${HOST_WORKDIR}}"
FIGURES_DIR_IN_CONTAINER="/work${FIGURES_DIR#${HOST_WORKDIR}}"

echo "Container will use:"
echo "  OUTPUT_DIR  -> ${OUTPUT_DIR_IN_CONTAINER}"
echo "  FIGURES_DIR -> ${FIGURES_DIR_IN_CONTAINER}"

# --- Run container & plotting script -----------------------------------------

docker run --rm -i \
  --user "$(id -u):$(id -g)" \
  -e OGGM_WORKDIR=/work \
  -e OUTPUT_DIR="${OUTPUT_DIR_IN_CONTAINER}" \
  -e FIGURES_DIR="${FIGURES_DIR_IN_CONTAINER}" \
  -e RUN_SCRIPTS_SUBDIR="${RUN_SCRIPTS_SUBDIR}" \
  -e PLOT_SCRIPT="${PLOT_SCRIPT}" \
  -v "${HOST_WORKDIR}":/work \
  "${AGILE_IMAGE}" \
  bash -s <<'EOF'
set -e

# Set a writable HOME under /work so libraries can store caches/configs
export HOME="${OGGM_WORKDIR}/fake_home_plot"
mkdir -p "${HOME}"

# Set up Matplotlib config/cache under HOME
export MPLCONFIGDIR="${HOME}/.config/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

# Set up salem (used by OGGM) cache under HOME
mkdir -p "${HOME}/.salem_cache"

# Create & activate plotting venv
rm -rf "${OGGM_WORKDIR}/plot_env"
python3 -m venv --system-site-packages "${OGGM_WORKDIR}/plot_env"
if [ ! -x "${OGGM_WORKDIR}/plot_env/bin/python3" ] && [ -x "${OGGM_WORKDIR}/plot_env/bin/python" ]; then
  ln -s python "${OGGM_WORKDIR}/plot_env/bin/python3"
fi
source "${OGGM_WORKDIR}/plot_env/bin/activate"

# Install packages
pip install --upgrade pip setuptools
pip install seaborn

# Change into directory with the plotting script
cd "/work/${RUN_SCRIPTS_SUBDIR}"

echo "Running plot script: ${PLOT_SCRIPT}"
echo "Using OUTPUT_DIR  = ${OUTPUT_DIR}"
echo "Writing figures to FIGURES_DIR = ${FIGURES_DIR}"

python3 "${PLOT_SCRIPT}"

EOF

echo
echo "Plotting finished."
echo "Figures written into: ${FIGURES_DIR}"

