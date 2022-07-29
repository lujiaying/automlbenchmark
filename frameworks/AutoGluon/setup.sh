#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/awslabs/autogluon.git"}
PKG=${3:-"autogluon"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true
#if [[ -x "$(command -v apt-get)" ]]; then
#    SUDO apt-get install -y libomp-dev
if [[ -x "$(command -v brew)" ]]; then
    brew install libomp
fi

PIP install --upgrade pip
PIP install --upgrade setuptools wheel
PIP install "mxnet<2.0.0"
PIP install "scikit-learn-intelex<2021.3"

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    cd ${TARGET_DIR}
    # PIP install -e core/
    # PIP install -e features/
    # PIP install -e tabular/[all]
    # PIP install -e mxnet/
    # PIP install -e extra/
    # PIP install -e text/
    # PIP install -e vision/
    # PIP install -e autogluon/
    PIP install torch==1.12.0+cpu torchvision==0.13.0+cpu torchtext==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
    set -euo pipefail
    PIP install -e common/[tests]
    PIP install -e core/[all,tests]
    PIP install -e features/
    PIP install -e tabular/[all,tests]
    PIP install -e multimodal/[tests]
    PIP install -e text/[tests]
    PIP install -e vision/
    PIP install -e timeseries/[all,tests]
    PIP install -e autogluon/
fi

PY -c "from autogluon.tabular.version import __version__; print(__version__)" >> "${HERE}/.setup/installed"
