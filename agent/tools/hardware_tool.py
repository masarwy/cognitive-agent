import os
import platform
import subprocess
import psutil

from agent.tools.base import Tool


class HardwareTool(Tool):

    def __init__(self):
        super().__init__("hardware_analyze")

    # ------------------------

    def execute(self, input_text: str = "") -> str:

        print("[HardwareTool] Detecting hardware...")

        info = {}

        info["os"] = platform.system()
        info["os_version"] = platform.version()
        info["architecture"] = platform.machine()
        info["python"] = platform.python_version()
        info["cpu"] = platform.processor()

        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)

        info["jetson"] = self._detect_jetson()
        info["gpu"] = self._detect_gpu()
        info["cuda"] = self._detect_cuda()
        info["tensorrt"] = self._detect_tensorrt()

        acceleration_info = self._detect_hardware_acceleration()
        info.update(acceleration_info)

        return self._format(info)

    # ------------------------

    def _detect_jetson(self):

        try:
            if os.path.exists("/etc/nv_tegra_release"):
                return True
        except:
            pass

        return False

    # ------------------------

    def _detect_gpu(self):

        try:

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:

                lines = result.stdout.strip().split("\n")

                gpus = []

                for line in lines:
                    name, memory = line.split(",")

                    gpus.append({
                        "name": name.strip(),
                        "memory": memory.strip()
                    })

                return gpus

        except:
            pass

        return "No NVIDIA GPU detected"

    # ------------------------

    def _detect_cuda(self):

        try:

            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return result.stdout.split("\n")[-2]

        except:
            pass

        return "CUDA not found"

    # ------------------------

    def _detect_hardware_acceleration(self):
        """Detect hardware-level acceleration capabilities"""
        info = {}

        # CUDA compute capability (if NVIDIA GPU present)
        if self._detect_cuda():
            try:
                import torch
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability()
                    info['compute_capability'] = f"{capability[0]}.{capability[1]}"

                    # Tensor cores (hardware feature)
                    if capability[0] >= 7:
                        info['tensor_cores'] = "Available"
            except:
                pass

        # Intel optimizations (CPU feature)
        if 'intel' in platform.processor().lower():
            info['cpu_vendor'] = "Intel"
            info['avx512'] = self._check_cpu_feature('avx512')
            info['avx2'] = self._check_cpu_feature('avx2')
        elif 'amd' in platform.processor().lower():
            info['cpu_vendor'] = "AMD"

        return info

    def _check_cpu_feature(self, feature):
        """Check if CPU supports specific instruction set"""
        try:
            # On Linux
            with open('/proc/cpuinfo') as f:
                flags = f.read()
                return "Supported" if feature in flags else "Not supported"
        except:
            return "Unknown"

    # ------------------------

    def _detect_tensorrt(self):

        try:

            import tensorrt as trt

            return trt.__version__

        except:
            return "TensorRT not found"

    # ------------------------

    def _format(self, info):

        lines = []

        lines.append("Hardware Profile:")
        lines.append("------------------")

        for key, value in info.items():
            lines.append(f"{key}: {value}")

        return "\n".join(lines)
