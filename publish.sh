#!/bin/bash
set -o errexit

# This script requires these tools:
# pip3 install --user --upgrade build twine

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

# clean up
rm -rf build
rm -rf dist
rm -rf spectralcluster.egg-info

# build and upload
python3 -m build
python3 -m twine upload dist/* --verbose

popd