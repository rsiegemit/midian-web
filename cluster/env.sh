#!/bin/bash
# Sourced by the cluster scripts: cluster/cluster.env (untracked; see cluster.env.example) if present, then
# `need VAR ...` fails with a named error for any unset variable.
_rte_cluster="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
[ -f "$_rte_cluster/cluster.env" ] && . "$_rte_cluster/cluster.env"
export RTE_REPO="${RTE_REPO:-$(dirname "$_rte_cluster")}"
need() {
  local v
  for v in "$@"; do [ -n "${!v:-}" ] || { echo "$v is not set (cluster/cluster.env.example)" >&2; exit 2; }; done
}
