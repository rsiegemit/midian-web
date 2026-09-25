#!/usr/bin/env bash
# Reproducible build of $RTE_DATA/env/fw_crewai (SPEC §6A). Login node only (needs internet).
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
fw_build_isolated crewai
