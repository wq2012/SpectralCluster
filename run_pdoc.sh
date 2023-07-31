#!/bin/bash
set -o errexit

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

rm -rf docs

# This script requires pdoc:
# pip3 install pdoc
pdoc spectralcluster -o docs

popd
