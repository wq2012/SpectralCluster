#!/bin/bash
set -o errexit

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

rm -r docs

# This script requires pdoc3 at least 0.5.2:
# pip3 install pdoc3
python3 -m pdoc spectralcluster --html --html-dir=docs

mv docs/spectralcluster/* docs/

rm -r docs/spectralcluster

popd
