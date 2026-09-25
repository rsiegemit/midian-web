"""Write configs/grid.yaml: the resolved configs/grids/ as ONE plain YAML file (no named sets), GENERATED.

It exists only for scripts that still read configs/grid.yaml with yaml.safe_load; rte.run reads configs/grids/ through
rte.run.load_config(). Regenerate after every edit under configs/grids/ (tests/test_load_config.py checks it is in
sync); delete it once no script opens configs/grid.yaml.

    PYTHONPATH=. python configs/export_grid_yaml.py
"""
import os

import yaml

from rte.run import load_config

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid.yaml")
HEADER = ("# GENERATED from configs/grids/ by configs/export_grid_yaml.py -- do not edit. Shared method lists\n"
          "# appear as YAML anchors (&idNNN) / aliases (*idNNN). Kept only for scripts that still read one file.\n")


def main():
    cfg = load_config()
    body = yaml.safe_dump({"defaults": cfg["defaults"], "grids": cfg["grids"]}, sort_keys=False, width=120,
                          default_flow_style=None)
    with open(OUT, "w") as fh:
        fh.write(HEADER + body)


if __name__ == "__main__":
    main()
