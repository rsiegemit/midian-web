#!/usr/bin/env bash
# Build $RTE_DATA/env/fw_maf (SPEC §6A). Login node only (needs internet).
export PYTHONNOUSERSITE=1     # ~/.local/lib/python3.12/site-packages is on every conda prefix's path and
                              # shadows deps (numpy, a `google` namespace pkg), so pip skips installing them
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
fw_build maf
