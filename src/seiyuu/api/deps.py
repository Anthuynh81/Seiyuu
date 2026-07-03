"""Request-scoped accessors over the lifespan-owned singletons on ``app.state``."""

from typing import Annotated

from fastapi import Depends, Request

from seiyuu.api.concurrency import HeavyWorkGate
from seiyuu.api.registry import EngineRegistry
from seiyuu.jobs import JobRunner
from seiyuu.repository import JobStore
from seiyuu.settings import Settings


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _store(request: Request) -> JobStore:
    return request.app.state.store


def _runner(request: Request) -> JobRunner:
    return request.app.state.runner


def _registry(request: Request) -> EngineRegistry:
    return request.app.state.registry


def _gate(request: Request) -> HeavyWorkGate:
    return request.app.state.gate


def _reconciled(request: Request) -> int:
    return request.app.state.reconciled_at_startup


SettingsDep = Annotated[Settings, Depends(_settings)]
StoreDep = Annotated[JobStore, Depends(_store)]
RunnerDep = Annotated[JobRunner, Depends(_runner)]
RegistryDep = Annotated[EngineRegistry, Depends(_registry)]
GateDep = Annotated[HeavyWorkGate, Depends(_gate)]
ReconciledDep = Annotated[int, Depends(_reconciled)]
