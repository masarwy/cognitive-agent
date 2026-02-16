import re
from typing import Dict, Any

from agent.tools.base import Tool


class CodeAnalysisTool(Tool):

    def __init__(self):
        super().__init__("code_analysis")

    # ==========================================================
    # Main Entry
    # ==========================================================

    def execute(self, input_str: str) -> str:

        print("[CodeAnalysisTool] Analyzing code...")

        if not input_str or not isinstance(input_str, str):
            raise ValueError("Invalid tool input")

        parsed = self._parse_input(input_str)

        if "code" not in parsed:
            raise ValueError("Invalid tool input")

        code = parsed.get("code")

        analysis = {
            "frameworks": self._detect_frameworks(code),
            "device_usage": self._detect_device(code),
            "precision": self._detect_precision(code),
            "batching": self._detect_batching(code),
            "camera_usage": self._detect_camera(code),
            "tensorrt_usage": self._detect_tensorrt(code),
            "inference_mode": self._detect_inference_mode(code),
            "cpu_preprocessing": self._detect_cpu_preprocessing(code),
            "training_code_detected": self._detect_training_code(code),
            "memory_fragmentation_risk": self._detect_fragmentation(code),
            "blocking_camera_loop": self._detect_blocking_camera(code),
        }

        analysis["optimization_score"] = self._score_pipeline(analysis)
        analysis["expected_speedup"] = self._estimate_speedup(analysis)
        analysis["optimization_score"] = self._score_pipeline(analysis)
        analysis["optimization_hints"] = self._generate_hints(analysis)

        return self._format(analysis)

    # ==========================================================
    # Detection Layer
    # ==========================================================

    def _detect_frameworks(self, code: str):

        code_lower = code.lower()
        frameworks = []

        if "torch" in code_lower:
            frameworks.append("pytorch")

        if "tensorrt" in code_lower or "trt." in code_lower:
            frameworks.append("tensorrt")

        if "onnx" in code_lower:
            frameworks.append("onnx")

        if "numpy" in code_lower:
            frameworks.append("numpy")

        if "cv2" in code_lower:
            frameworks.append("opencv")

        return frameworks

    # ----------------------------------------------------------

    def _detect_device(self, code: str):

        code_lower = code.lower()

        if ".cuda(" in code_lower or "to('cuda')" in code_lower or 'to("cuda")' in code_lower:
            return "cuda"

        if ".cpu(" in code_lower or "device='cpu'" in code_lower:
            return "cpu"

        return "unknown"

    # ----------------------------------------------------------

    def _detect_precision(self, code: str):

        code_lower = code.lower()

        if "float16" in code_lower or "fp16" in code_lower or ".half()" in code_lower:
            return "fp16"

        if "int8" in code_lower:
            return "int8"

        if "float32" in code_lower or "fp32" in code_lower:
            return "fp32"

        return "unknown"

    # ----------------------------------------------------------

    def _detect_batching(self, code: str):

        matches = re.findall(r"batch[_ ]?size\s*=\s*(\d+)", code.lower())
        if matches:
            return int(matches[0])

        return None

    # ----------------------------------------------------------

    def _detect_camera(self, code: str):

        code_lower = code.lower()

        return (
                "cv2.videocapture" in code_lower
                or "gstreamer" in code_lower
                or "nvarguscamerasrc" in code_lower
        )

    # ----------------------------------------------------------

    def _detect_tensorrt(self, code: str):

        code_lower = code.lower()
        return "tensorrt" in code_lower or "trt." in code_lower

    # ----------------------------------------------------------

    def _detect_inference_mode(self, code: str):

        code_lower = code.lower()

        if "torch.no_grad()" in code_lower:
            return True

        if "torch.inference_mode()" in code_lower:
            return True

        return False

    # ----------------------------------------------------------

    def _detect_cpu_preprocessing(self, code: str):

        code_lower = code.lower()

        if "numpy" in code_lower and "cuda" not in code_lower:
            return True

        if "cv2.resize" in code_lower and "cuda" not in code_lower:
            return True

        return False

    # ----------------------------------------------------------

    def _detect_training_code(self, code):

        code_lower = code.lower()

        training_patterns = [
            "optimizer.step(",
            "loss.backward(",
            ".backward(",
            "model.train(",
            "requires_grad",
            "zero_grad("
        ]

        for pattern in training_patterns:
            if pattern in code_lower:
                return True

        return False

    # ----------------------------------------------------------

    def _detect_fragmentation(self, code):

        code_lower = code.lower()

        risky_patterns = [
            ".cuda()",  # repeated cuda transfers
            ".to('cuda')",
            ".to(\"cuda\")",
            "torch.tensor(",
            "torch.from_numpy("
        ]

        count = 0

        for pattern in risky_patterns:
            count += code_lower.count(pattern)

        if count > 5:
            return "high"

        if count > 2:
            return "medium"

        return "low"

    # ----------------------------------------------------------

    def _detect_blocking_camera(self, code):

        code_lower = code.lower()

        if "while true" in code_lower or "while(True)" in code_lower:

            if "videocapture.read" in code_lower:

                if "thread" not in code_lower and "queue" not in code_lower:
                    return True

        return False

    # ==========================================================
    # Optimization Intelligence
    # ==========================================================

    def _score_pipeline(self, analysis: Dict[str, Any]) -> int:

        score = 100

        if analysis["device_usage"] != "cuda":
            score -= 25

        if analysis["precision"] == "fp32":
            score -= 15

        if not analysis["tensorrt_usage"]:
            score -= 20

        if analysis["batching"] is None:
            score -= 10

        if not analysis["inference_mode"]:
            score -= 10

        if analysis["cpu_preprocessing"]:
            score -= 10

        return max(score, 0)

    # ----------------------------------------------------------

    def _generate_hints(self, analysis):

        hints = []

        if analysis["device_usage"] != "cuda":
            hints.append("Enable CUDA for GPU acceleration.")

        if analysis["precision"] == "fp32":
            hints.append("Switch to FP16 or INT8 for faster inference.")

        if not analysis["tensorrt_usage"]:
            hints.append("Convert model to TensorRT for significant speedup.")

        if analysis["batching"] is None:
            hints.append("Introduce batching to increase throughput.")

        if not analysis["inference_mode"]:
            hints.append("Wrap inference with torch.no_grad() or inference_mode().")

        if analysis["cpu_preprocessing"]:
            hints.append("Move preprocessing to GPU to avoid CPU bottlenecks.")

        if analysis["camera_usage"]:
            hints.append("Use zero-copy or GPU-accelerated camera pipelines on Jetson.")

        if analysis["training_code_detected"]:
            hints.append("Training code detected. Disable gradients for inference.")

        if analysis["memory_fragmentation_risk"] == "high":
            hints.append("High CUDA memory fragmentation risk detected.")

        if analysis["blocking_camera_loop"]:
            hints.append("Camera loop is blocking. Use threaded or async pipeline.")

        if analysis["expected_speedup"] > 2:
            hints.append(
                f"Estimated speedup potential: {analysis['expected_speedup']}x"
            )

        return hints

    # ----------------------------------------------------------

    def _estimate_speedup(self, analysis):

        multiplier = 1.0

        if analysis["device_usage"] != "cuda":
            multiplier *= 5.0

        if analysis["precision"] == "fp32":
            multiplier *= 1.8

        if not analysis["tensorrt_usage"]:
            multiplier *= 2.5

        if analysis["batching"] is None:
            multiplier *= 1.5

        if analysis["cpu_preprocessing"]:
            multiplier *= 1.3

        return round(multiplier, 2)

    # ==========================================================
    # Formatting
    # ==========================================================

    def _format(self, info):

        lines = [
            "CodeAnalysis:",
            "------------------"
        ]

        for key, value in info.items():
            lines.append(f"{key}: {value}")

        return "\n".join(lines)

    # ==========================================================
    # Input Parser
    # ==========================================================

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
