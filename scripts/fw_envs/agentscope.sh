#!/usr/bin/env bash
# Build $RTE_DATA/env/fw_agentscope (SPEC §6A). Run on a login node (needs internet).
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
fw_build agentscope
