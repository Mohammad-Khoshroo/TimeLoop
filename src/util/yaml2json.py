#!/usr/bin/env python3
# ==============================================================================
# ابزار تبدیل خودکار دوطرفه میان YAML و JSON برای پروژه Timeloop + Accelergy
# Bidirectional YAML <-> JSON Converter for Timeloop & Accelergy
# ==============================================================================

import sys
import json
import yaml
from pathlib import Path

def yaml_to_json(yaml_path: Path, json_path: Path = None) -> Path:
    if json_path is None:
        json_path = yaml_path.with_suffix(".json")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return json_path

def json_to_yaml(json_path: Path, yaml_path: Path = None) -> Path:
    if yaml_path is None:
        yaml_path = json_path.with_suffix(".yaml")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
    return yaml_path

def main():
    if len(sys.argv) < 2:
        print("Usage: yaml2json.py <file.yaml|file.json> [output_file]")
        sys.exit(1)
        
    src_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if src_path.suffix.lower() in [".yaml", ".yml"]:
        res = yaml_to_json(src_path, out_path)
        print(f"Converted YAML -> JSON: {res}")
    elif src_path.suffix.lower() == ".json":
        res = json_to_yaml(src_path, out_path)
        print(f"Converted JSON -> YAML: {res}")
    else:
        print(f"Error: Unknown extension for {src_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
