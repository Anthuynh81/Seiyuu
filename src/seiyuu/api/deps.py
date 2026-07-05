"""Request-scoped accessors over the lifespan-owned singletons on ``app.state``."""

import threading
from typing import Annotated

from fastapi import Depends, Request

from seiyuu.api.concurrency import BorrowBroker, HeavyWorkGate
from seiyuu.api.registry import EngineRegistry
from seiyuu.jobs import JobRunner
from seiyuu.repository import JobStore
from seiyuu.settings import Settings
from seiyuu.validate import Validator


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


def _broker(request: Request) -> BorrowBroker:
    return request.app.state.borrow_broker


def _reconciled(request: Request) -> int:
    return request.app.state.reconciled_at_startup


def _aligner(request: Request) -> tuple[Validator, threading.Lock]:
    """The process-shared CPU whisper aligner + its serialization lock (F2). One Validator for
    the whole process (lazy model load) and one lock, so every forced-alignment request
    serializes — CTranslate2 is not safe under concurrent transcribe — and alignment never
    spins up a second whisper model per request."""
    return request.app.state.aligner, request.app.state.align_lock


SettingsDep = Annotated[Settings, Depends(_settings)]
StoreDep = Annotated[JobStore, Depends(_store)]
RunnerDep = Annotated[JobRunner, Depends(_runner)]
RegistryDep = Annotated[EngineRegistry, Depends(_registry)]
GateDep = Annotated[HeavyWorkGate, Depends(_gate)]
BrokerDep = Annotated[BorrowBroker, Depends(_broker)]
ReconciledDep = Annotated[int, Depends(_reconciled)]
AlignerDep = Annotated[tuple[Validator, threading.Lock], Depends(_aligner)]
