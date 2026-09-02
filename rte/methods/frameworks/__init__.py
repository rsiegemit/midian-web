"""Framework rivals (SPEC §6A): the real agent-management libraries, each run in its OWN venv
(`$RTE_DATA/env/fw_<name>`) as a JSON-lines worker subprocess, driven by a thin Method here.
Shared pieces: `_common.FrameworkMethod` (retrieval adapter + ledger + agent-name mapping) and
`_bridge.Bridge` (subprocess protocol). One `fw_<name>.py` per framework, ≤80 lines each."""
