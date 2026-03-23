"""LangChain engineering templates with high-level app facade."""

from importlib.metadata import PackageNotFoundError, version

from lc_templates.app import TemplateApp, create_app
from lc_templates.core.config import scaffold_config
from lc_templates.core.hooks import clear_event_hooks, register_event_hook, unregister_event_hook
from lc_templates.core.schemas import (
    AgentExecutionResult,
    ClassificationResult,
    ExtractionResult,
    GroundedAnswer,
    HealthReport,
    HookEvent,
    KnowledgeBaseBuildResult,
    MemoryThreadOperationResult,
    RouteDecision,
    TaskBundleResult,
    WorkflowExecutionResult,
)

try:
    __version__ = version("langchain12-templates")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "AgentExecutionResult",
    "ClassificationResult",
    "ExtractionResult",
    "GroundedAnswer",
    "HealthReport",
    "HookEvent",
    "KnowledgeBaseBuildResult",
    "MemoryThreadOperationResult",
    "RouteDecision",
    "TaskBundleResult",
    "TemplateApp",
    "WorkflowExecutionResult",
    "__version__",
    "clear_event_hooks",
    "create_app",
    "register_event_hook",
    "scaffold_config",
    "unregister_event_hook",
]
