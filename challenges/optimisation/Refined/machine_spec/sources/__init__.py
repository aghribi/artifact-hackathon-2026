"""Machine-specific readers, for authoring a spec rather than running one.

A calibration table, a magnet archive, a naming convention: the things one
accelerator knows about itself. They live here and not in the environment,
because they are how a `MachineSpec` gets *written*, and nothing that runs a
machine should need them.
"""
