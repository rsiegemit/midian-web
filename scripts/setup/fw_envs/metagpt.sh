#!/usr/bin/env bash
# Build $RTE_DATA/env/fw_metagpt (SPEC §6A appendix). metagpt 0.8.2 requires python <3.12,
# so the prefix is created with 3.11 here; fw_build then only pip-installs into it.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
prefix="$RTE_DATA/env/fw_metagpt"
if [ ! -x "$prefix/bin/python" ]; then
  fw_conda
  conda create -y -p "$prefix" python=3.11
fi
fw_build metagpt
