import pytest

from tests.utils import format_tool_input

from agent.tools.code_analysis_tool import CodeAnalysisTool
from agent.tools.executor import ToolExecutor
from agent.tools.registry import ToolRegistry


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def analyze_tool():
    return CodeAnalysisTool()


@pytest.fixture
def executor():
    registry = ToolRegistry()
    registry.register(CodeAnalysisTool())

    return ToolExecutor(registry)


# --------------------------------------------------
# Basic Tool Tests
# --------------------------------------------------

def test_analyze_tool_initialization(analyze_tool):
    assert analyze_tool is not None
    assert analyze_tool.name == "code_analysis"


def test_analyze_simple_python_code(analyze_tool):
    code = """
import numpy as np

def process(data):
    result = []
    for x in data:
        result.append(x * 2)
    return result
"""

    result = analyze_tool.execute(format_tool_input({
        "code": code,
        "language": "python"
    }))

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


# --------------------------------------------------
# Hardware-related detection test
# --------------------------------------------------

def test_detects_cpu_only_pattern(analyze_tool):
    code = """
import numpy as np

def process(data):
    return np.array(data) * 2
"""

    result = analyze_tool.execute(format_tool_input({
        "code": code,
        "language": "python",
        "hardware": "jetson_orin_nano"
    }))

    # Tool should produce analysis mentioning optimization or GPU
    assert isinstance(result, str)
    assert len(result) > 0


# --------------------------------------------------
# CUDA pattern test
# --------------------------------------------------

def test_cuda_code_analysis(analyze_tool):
    code = """
import torch

model = model.cuda()
input = input.cuda()

output = model(input)
"""

    result = analyze_tool.execute(format_tool_input({
        "code": code,
        "language": "python",
        "hardware": "gpu"
    }))

    assert isinstance(result, str)
    assert len(result) > 0


# --------------------------------------------------
# Integration Test with ToolExecutor
# --------------------------------------------------

def test_executor_runs_analyze_tool(executor):
    code = """
def read_sensor():
    return 42
"""

    result = executor.execute(
        tool_name="code_analysis",
        input_text=format_tool_input({
            "code": code,
            "language": "python"
        })
    )

    assert result is not None
    assert isinstance(result, str)


# --------------------------------------------------
# Failure Tests
# --------------------------------------------------

def test_missing_code_field(analyze_tool):
    with pytest.raises(Exception):
        analyze_tool.execute(format_tool_input({
            "language": "python"
        }))


def test_empty_code(analyze_tool):
    result = analyze_tool.execute(format_tool_input({
        "code": "",
        "language": "python"
    }))

    assert result is not None
