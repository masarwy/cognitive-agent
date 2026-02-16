from typing import Dict, Any

from agent.tools.base import Tool
from agent.tools.hardware_tool import HardwareTool
from agent.tools.code_analysis_tool import CodeAnalysisTool


class OptimizationTool(Tool):

    def __init__(self):
        super().__init__("optimize")

        self.hardware_tool = HardwareTool()
        self.code_tool = CodeAnalysisTool()

    # ---------------------------------------------------------

    def execute(self, input_str: str) -> str:

        print("[OptimizationTool] Generating optimization plan...")

        if not input_str or not isinstance(input_str, str):
            raise ValueError("Invalid tool input")

        parsed = self._parse_input(input_str)

        if "code" not in parsed:
            raise ValueError("Code not provided")

        code = parsed["code"]

        # Step 1 — detect hardware
        hw_info_str = self.hardware_tool.execute("")
        hw_info = self._parse_kv(hw_info_str)

        # Step 2 — analyze code
        code_analysis_str = self.code_tool.execute(input_str)
        code_info = self._parse_kv(code_analysis_str)

        # Step 3 — generate optimization plan
        plan = self._build_plan(hw_info, code_info)

        return self._format(plan)

    # ---------------------------------------------------------

    def _build_plan(self, hw: Dict, code: Dict) -> Dict:

        actions = []
        expected_speedup = code.get("expected_speedup", "unknown")

        gpu = hw.get("gpu", "unknown")
        ram = hw.get("ram", "unknown")

        device = code.get("device_usage", "unknown")
        precision = code.get("precision", "unknown")
        tensorrt = code.get("tensorrt_usage", "False")
        batching = code.get("batching", "None")
        fragmentation = code.get("memory_fragmentation_risk", "low")

        # GPU optimization
        if device != "cuda" and gpu != "none":
            actions.append({
                "priority": "HIGH",
                "action": "Enable CUDA acceleration",
                "impact": "5–20x speedup"
            })

        # Precision optimization
        if precision == "fp32":
            actions.append({
                "priority": "HIGH",
                "action": "Convert inference to FP16",
                "impact": "1.5–3x speedup"
            })

        # TensorRT optimization
        if tensorrt == "False":
            actions.append({
                "priority": "CRITICAL",
                "action": "Convert model to TensorRT",
                "impact": "2–5x speedup"
            })

        # batching
        if batching == "None":
            actions.append({
                "priority": "MEDIUM",
                "action": "Introduce batching",
                "impact": "1.5–4x throughput increase"
            })

        # fragmentation
        if fragmentation == "high":
            actions.append({
                "priority": "HIGH",
                "action": "Fix CUDA memory fragmentation",
                "impact": "Prevents crashes and slowdowns"
            })

        return {
            "hardware": hw,
            "code": code,
            "estimated_speedup": expected_speedup,
            "recommended_actions": actions
        }

    # ---------------------------------------------------------

    def _format(self, plan: Dict) -> str:

        lines = []
        lines.append("Optimization Plan")
        lines.append("=================")

        lines.append("\nEstimated Speedup:")
        lines.append(f"{plan['estimated_speedup']}x")

        lines.append("\nRecommended Actions:")

        for action in plan["recommended_actions"]:

            lines.append("")
            lines.append(f"[{action['priority']}] {action['action']}")
            lines.append(f"Impact: {action['impact']}")

        return "\n".join(lines)

    # ---------------------------------------------------------

    def _parse_input(self, input_str: str) -> dict:

        result = {}
        lines = input_str.splitlines()

        key = None
        buffer = []

        for line in lines:

            if line.endswith(": |"):
                if key:
                    result[key] = "\n".join(buffer)
                    buffer = []

                key = line.split(":")[0].strip()

            elif key and line.startswith("  "):
                buffer.append(line[2:])

            elif ":" in line:
                if key:
                    result[key] = "\n".join(buffer)
                    buffer = []
                    key = None

                k, v = line.split(":", 1)
                result[k.strip()] = v.strip()

        if key:
            result[key] = "\n".join(buffer)

        return result

    # ---------------------------------------------------------

    def _parse_kv(self, text: str) -> Dict:

        result = {}

        for line in text.splitlines():

            if ":" in line:
                k, v = line.split(":", 1)
                result[k.strip()] = v.strip()

        return result
