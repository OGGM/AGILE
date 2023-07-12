###################################################
# Dockerfile to build a Python environment
# with OGGM installed, based on latest Ubuntu.
###################################################

FROM ghcr.io/oggm/oggm:latest

ARG GITHUB_SHA=master
ARG GITHUB_REPOSITORY=OGGM/agile
RUN $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN $PIP install "git+https://github.com/${GITHUB_REPOSITORY}@${GITHUB_SHA}"

ADD test.sh /root/test.sh
