import argparse
import csv
import json
import sys

import yaml

from polyevo.config import load_config
from polyevo.core.engine import run_ga
from polyevo.core.registry import list_ops
from polyevo.encoding.library import build_library


def main(argv=None):
    parser = argparse.ArgumentParser(prog="polyevo", description="PolyEvo M1 evolutionary search")
    commands = parser.add_subparsers(dest="command", required=True)
    validate_parser = commands.add_parser("validate", help="Validate hooks and print resolved YAML")
    validate_parser.add_argument("config")
    run_parser = commands.add_parser("run", help="Run a configured search")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--out", required=True)
    ops_parser = commands.add_parser("list-ops", help="List implemented registry entries")
    ops_parser.add_argument("--stage")
    library_parser = commands.add_parser("build-library", help="Split polymer CSV into a BRICS fragment library")
    library_parser.add_argument("--input", required=True)
    library_parser.add_argument("--out", required=True)
    library_parser.add_argument("--smiles-column", default="smiles")
    library_parser.add_argument("--min-heavy-atoms", type=int, default=1)
    library_parser.add_argument("--max-heavy-atoms", type=int)
    library_parser.add_argument("--forbid-smarts", action="append", default=[])
    library_parser.add_argument("--allowed-elements", nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            print(yaml.safe_dump(load_config(args.config), sort_keys=False), end="")
        elif args.command == "run":
            state = run_ga(load_config(args.config), args.out)
            print(json.dumps({"out": str(state.out), "generation": state.generation,
                              "oracle_calls": state.oracle_calls, "stop_reason": state.stop_reason}))
        elif args.command == "build-library":
            report = build_library(args.input, args.out, smiles_column=args.smiles_column,
                                   min_heavy_atoms=args.min_heavy_atoms,
                                   max_heavy_atoms=args.max_heavy_atoms,
                                   forbidden_smarts=args.forbid_smarts,
                                   allowed_elements=args.allowed_elements)
            print(json.dumps(report, ensure_ascii=False))
        else:
            print(yaml.safe_dump(list_ops(args.stage), sort_keys=False), end="")
    except (ValueError, TypeError, KeyError, OSError, csv.Error, yaml.YAMLError) as exc:
        print(f"polyevo: {exc}", file=sys.stderr)
        return 2
    return 0
