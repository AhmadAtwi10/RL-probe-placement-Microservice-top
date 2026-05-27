"""
src/simulator/workload
----------------------
Public API for the workload layer (Phase 2).

Typical usage in the environment:

    from src.simulator.workload import SLIGenerator, FailureInjector

    generator = SLIGenerator(ep_graph, episode_length=100, seed=42)
    injector  = FailureInjector(ep_graph, episode_length=100, seed=42)

    traces = generator.full_traces()
    events = injector.sample_failures(n_failures=2)
    traces = injector.apply(traces, events)

    snapshot = {node: {m: traces[node][m][t] for m in traces[node]}
                for node in traces}
"""

from src.simulator.workload.sli_generator import (
    SLIGenerator,
    BASELINES,
    NOISE_STD,
)

from src.simulator.workload.failure_injector import (
    FailureInjector,
    FailureEvent,
    FailureMode,
    PROPAGATION_FACTOR,
    DEFAULT_RAMP_STEPS,
)

__all__ = [
    "SLIGenerator",
    "BASELINES",
    "NOISE_STD",
    "FailureInjector",
    "FailureEvent",
    "FailureMode",
    "PROPAGATION_FACTOR",
    "DEFAULT_RAMP_STEPS",
]
