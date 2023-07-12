#!/bin/bash
set -e
set -x

chown -R "$(id -u):$(id -g)" "$HOME"

export TEST_ENV="$1"

[[ -d .git ]] || export SETUPTOOLS_SCM_PRETEND_VERSION="g$GITHUB_SHA"

$PIP install --upgrade coverage coveralls git+https://github.com/fmaussion/salem.git
$PIP install -e .

export COVERAGE_RCFILE="$PWD/.coveragerc"

coverage erase

coverage run --source=./agile1d --parallel-mode --module \
    pytest --verbose --run-test-env $TEST_ENV agile1d

coverage agile
coverage xml
coverage report --skip-covered
