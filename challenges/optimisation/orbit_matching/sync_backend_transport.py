"""Extract SimEnv's transport adapters without its optional Gym environment dependency."""
import ast
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    source=(root/'RL_framework/rl_framework/envs/sim_env.py').read_text()
    tree=ast.parse(source)
    header='''"""Generated from RL_framework SimEnv transport classes; run sync_backend_transport to refresh.

Physics is supplied by nominal_model; this avoids requiring Gym for batch studies.
"""
from __future__ import annotations
from abc import ABC,abstractmethod
from typing import Iterator
import numpy as np
import torch
import cheetah
DTYPE=torch.float32
'''
    parts=[header]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VBC_EDGE' for t in node.targets):parts.append(ast.get_source_segment(source,node))
        if isinstance(node,ast.ClassDef) and node.name in ['Transport','CheetahTransport','XsuiteTransport']:parts.append(ast.get_source_segment(source,node))
    (root/'orbit_matching/backend_transport.py').write_text('\n\n'.join(parts)+'\n')

if __name__=='__main__':main()
