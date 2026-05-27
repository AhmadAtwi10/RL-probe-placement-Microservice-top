"""Quick validation script for node_config.py"""
import sys
sys.path.insert(0, '.')

from src.simulator.config.node_config import (
    NODE_CATALOG, CORE_NODES, NON_CORE_NODES,
    PROBEABLE_NODES, INFRA_NODES, ALL_NODES
)

print(f"Total nodes    : {len(ALL_NODES)}")
print(f"Core nodes     : {len(CORE_NODES)}")
print(f"Non-core nodes : {len(NON_CORE_NODES)}")
print(f"Probeable nodes: {len(PROBEABLE_NODES)}")
print(f"Infra nodes    : {INFRA_NODES}")

assert len(ALL_NODES) == 24,       f"Expected 24 nodes, got {len(ALL_NODES)}"
assert len(CORE_NODES) == 10,      f"Expected 10 core nodes, got {len(CORE_NODES)}"
assert len(NON_CORE_NODES) == 14,  f"Expected 14 non-core nodes, got {len(NON_CORE_NODES)}"
assert len(PROBEABLE_NODES) == 22, f"Expected 22 probeable nodes, got {len(PROBEABLE_NODES)}"
assert len(INFRA_NODES) == 2,      f"Expected 2 infra nodes, got {len(INFRA_NODES)}"

# Check one-hot encoding works
meta = NODE_CATALOG["frontend"]
oh = meta.type_one_hot()
assert sum(oh) == 1, "One-hot must have exactly one 1"
assert len(oh) == 7, "One-hot length must match NODE_TYPES"

print("\nAll assertions passed ✓")
