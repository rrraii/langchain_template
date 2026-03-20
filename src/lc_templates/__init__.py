"""LangChain engineering templates with high-level app facade."""

from importlib.metadata import PackageNotFoundError, version

from lc_templates.app import TemplateApp, create_app

try:
    __version__ = version("langchain12-templates")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["TemplateApp", "__version__", "create_app"]
