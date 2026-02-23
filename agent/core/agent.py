import re
from agent.planning.llm_planner import LLMPlanner
from agent.tools.registry import ToolRegistry
from agent.tools.executor import ToolExecutor
from agent.tools.retrieve_tool import RetrieveTool
from agent.tools.ingest_tool import IngestTool
from agent.tools.summarize_tool import SummarizeTool
from agent.tools.reason_tool import ReasonTool
from agent.tools.code_tool import CodeGenTool
from agent.tools.hardware_tool import HardwareTool
from agent.tools.code_analyze_tool import CodeAnalyzeTool
from agent.tools.github_clone_tool import GitHubCloneTool


class Agent:

    def __init__(self, name: str):
        self.name = name
        self.planner = LLMPlanner()
        self.registry = ToolRegistry()
        self._register_tools()
        self.executor = ToolExecutor(self.registry)
        self.full_context = True

    def _register_tools(self):
        self.registry.register(GitHubCloneTool())
        self.registry.register(RetrieveTool())
        self.registry.register(IngestTool())
        self.registry.register(SummarizeTool())
        self.registry.register(ReasonTool())
        self.registry.register(CodeGenTool())
        self.registry.register(HardwareTool())
        self.registry.register(CodeAnalyzeTool())

    def think(self, task: str, log_fn=None):
        def log(msg: str):
            if log_fn:
                log_fn(msg)
            else:
                print(msg)

        log(f"\n[{self.name}] Task:")
        log(task)

        plan = self.planner.create_plan(task)

        log(f"\n[{self.name}] Plan:")
        for step in plan:
            log(f"{step.id}. {step.description} (tool={step.tool})")

        return plan

    def execute(self, plan, log_fn=None):
        def log(msg: str):
            if log_fn:
                log_fn(msg)
            else:
                print(msg)

        log(f"\n[{self.name}] Executing:")

        step_results = {}
        failed_steps = set()

        for step in plan:
            log(f"\nStep {step.id}: {step.description}")

            # Fix path references dynamically based on previous results
            fixed_description = self._fix_path_references(step.description, step_results)

            if fixed_description != step.description:
                log(f"[PATH_FIX] Updated path in step description")
                log(f"  Original: {step.description}")
                log(f"  Fixed: {fixed_description}")
                step.description = fixed_description

            # Check dependencies before executing
            if self._check_dependencies(step, step_results, failed_steps):
                log(f"[SKIPPED] Step {step.id} skipped due to failed dependencies")
                failed_steps.add(step.id)
                continue

            # Build contextual input
            tool_input = self._build_tool_input(
                step_description=step.description,
                step_results=step_results,
                current_step_id=step.id
            )

            # Execute tool
            result = self.executor.execute(step.tool, tool_input)

            # Validate result with enhanced checking
            is_valid = self._validate_result(result, step)

            if not is_valid:
                log(f"[WARNING] Step {step.id} produced unexpected result")
                failed_steps.add(step.id)

            # Store result
            step_results[step.id] = {
                "tool": step.tool,
                "description": step.description,
                "result": result,
                "success": is_valid
            }

            log(f"Result: {result}")

        return step_results

    def _fix_path_references(self, description: str, step_results: dict) -> str:
        """
        Fix generic path references with actual paths from previous steps.
        Handles both explicit placeholders and missing paths.
        """
        actual_paths = {}

        for step_id, step_info in step_results.items():
            if step_info['tool'] == 'github_clone':
                result = step_info['result']

                patterns = [
                    r'(?:Repository is located at|cloned.*to):\s*(/[^\s\n]+)',
                    r'cloned\s+\w+\s+to\s+(/[^\s\n]+)',
                    r'(/tmp/[\w-]+)'
                ]

                for pattern in patterns:
                    match = re.search(pattern, result)
                    if match:
                        actual_paths['cloned_repo'] = match.group(1)
                        break

        if 'cloned_repo' not in actual_paths:
            return description

        actual_path = actual_paths['cloned_repo']

        # Only fix descriptions that actually need paths
        needs_path = any(keyword in description.lower() for keyword in
                         ['ingest', 'analyze code', 'retrieve', 'summarize code'])

        if not needs_path:
            return description

        generic_patterns = [
            (r'/path/to/cloned/[\w-]+', actual_path),
            (r'/home/user/[\w-]+', actual_path),
            (r'~/[\w-]+', actual_path),
        ]

        for pattern, replacement in generic_patterns:
            description = re.sub(pattern, replacement, description)

        if not re.search(r'/[\w/-]+', description):
            if re.search(r'\bingest\b', description, re.IGNORECASE):
                description = re.sub(
                    r'ingest\s+(cloned\s+)?(repository|directory|folder|data)',
                    f"ingest folder '{actual_path}'",
                    description,
                    flags=re.IGNORECASE
                )
            elif re.search(r'\banalyze\b.*\bcode\b', description, re.IGNORECASE):
                description = re.sub(
                    r'analyze\s+code(\s+for.*)?(\s+in)?(\s+ingested)?(\s+data)?',
                    f"analyze code in '{actual_path}' for memory usage",
                    description,
                    flags=re.IGNORECASE
                )

        return description

    def _build_tool_input(self, step_description: str, step_results: dict, current_step_id: int) -> str:
        """
        Build tool input as a string that includes context from previous steps.
        Uses smart truncation to keep context relevant and concise.
        """
        if not step_results:
            return step_description

        if self.full_context:
            context_parts = [
                f"Current Task: {step_description}",
                "\n--- Context from Previous Steps ---\n"
            ]

            for step_id in sorted(step_results.keys()):
                if step_id < current_step_id:
                    step_info = step_results[step_id]
                    result = step_info['result']

                    truncated_result = self._smart_truncate(
                        result=result,
                        tool_name=step_info['tool']
                    )

                    context_parts.append(
                        f"Step {step_id} ({step_info['tool']}): {step_info['description']}\n"
                        f"Result: {truncated_result}\n"
                    )

            return "\n".join(context_parts)
        else:
            prev_step_id = current_step_id - 1
            if prev_step_id in step_results:
                prev_result = step_results[prev_step_id]["result"]
                truncated = self._smart_truncate(
                    result=prev_result,
                    tool_name=step_results[prev_step_id]["tool"],
                    max_length=300
                )
                return f"{step_description}\n\nPrevious result:\n{truncated}"

            return step_description

    def _smart_truncate(self, result: str, tool_name: str, max_length: int = None) -> str:
        """
        Intelligently truncate results based on tool type.
        Preserves structure and important information.
        """
        if tool_name in ('ingest', 'github_clone', 'hardware_analyze'):
            return result

        if tool_name == 'code_analyze':
            return self._truncate_code_analysis(result, max_length or 800)

        if tool_name == 'reason':
            return self._truncate_reasoning(result, max_length or 800)

        default_max = max_length or 500
        if len(result) <= default_max:
            return result
        return result[:default_max] + f"... [truncated {len(result) - default_max} chars]"

    def _truncate_code_analysis(self, text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text

        lines = text.split('\n')
        kept_lines = []
        current_length = 0
        items_in_section = 0
        max_items_per_section = 3

        for line in lines:
            line_length = len(line) + 1

            if line.startswith('#'):
                kept_lines.append(line)
                current_length += line_length
                items_in_section = 0

            elif line.strip().startswith(('-', '*', '•', '**')):
                if items_in_section < max_items_per_section:
                    kept_lines.append(line)
                    current_length += line_length
                    items_in_section += 1
                elif items_in_section == max_items_per_section:
                    kept_lines.append("  [... more items ...]")
                    current_length += 25
                    items_in_section += 1

            elif current_length < max_length * 0.9:
                if line.strip() or current_length < max_length * 0.7:
                    kept_lines.append(line)
                    current_length += line_length

            if current_length >= max_length:
                break

        result = '\n'.join(kept_lines)
        if len(text) > len(result):
            result += f"\n\n[Analysis truncated: {len(text) - len(result)} chars omitted]"

        return result

    def _truncate_reasoning(self, text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text

        lines = text.split('\n')
        kept_lines = []
        current_length = 0
        in_summary = False

        for line in lines:
            line_length = len(line) + 1

            if 'executive summary' in line.lower() or 'summary' in line.lower():
                in_summary = True
                kept_lines.append(line)
                current_length += line_length
                continue

            if in_summary and not line.startswith('#'):
                kept_lines.append(line)
                current_length += line_length
                continue
            elif in_summary and line.startswith('#'):
                in_summary = False

            if line.startswith('#'):
                if current_length < max_length * 0.8:
                    kept_lines.append(line)
                    current_length += line_length
            elif line.strip().startswith(('-', '*', '•')) and current_length < max_length * 0.9:
                kept_lines.append(line)
                current_length += line_length

            if current_length >= max_length:
                break

        result = '\n'.join(kept_lines)
        if len(text) > len(result):
            result += f"\n\n[Reasoning truncated: full analysis available if needed]"

        return result

    def _validate_result(self, result: str, step) -> bool:
        if not result:
            print(f"[ERROR] Step {step.id} ({step.tool}) returned empty result")
            return False

        result_lower = result.lower()

        if step.tool == "ingest":
            return self._validate_ingest(result, result_lower, step)
        elif step.tool == "code_analyze":
            return self._validate_code_analyze(result, result_lower, step)
        elif step.tool == "hardware_analyze":
            return self._validate_hardware_analyze(result, result_lower, step)
        elif step.tool == "reason":
            return self._validate_reason(result, result_lower, step)
        else:
            return self._validate_generic(result, result_lower, step)

    def _validate_ingest(self, result: str, result_lower: str, step) -> bool:
        if "successfully ingested" not in result_lower:
            print(f"[ERROR] Ingest validation failed - expected success message")
            print(f"[RECOVERY] Check if RAG service is running and folder path is correct")
            return False

        match = re.search(r'(\d+)\s+documents?', result)
        if match:
            doc_count = int(match.group(1))
            if doc_count == 0:
                print(f"[ERROR] No documents ingested")
                print(f"[RECOVERY] Verify folder path exists and contains supported files")
                return False
            print(f"[INFO] Successfully ingested {doc_count} documents")

        return True

    def _validate_code_analyze(self, result: str, result_lower: str, step) -> bool:
        if "no code found" in result_lower:
            print(f"[ERROR] Code analysis failed - no code found")
            print(f"[RECOVERY] Ensure ingest step completed successfully before analysis")
            return False

        has_structure = any(marker in result for marker in ['##', '###', '**', '- '])
        if not has_structure:
            print(f"[WARNING] Code analysis output may be malformed")
            return False

        expected_sections = ['memory usage', 'issues', 'optimization']
        found_sections = sum(1 for section in expected_sections if section in result_lower)

        if found_sections < 2:
            print(f"[WARNING] Code analysis missing expected sections (found {found_sections}/3)")
            return False

        return True

    def _validate_hardware_analyze(self, result: str, result_lower: str, step) -> bool:
        required_fields = ['ram_gb', 'gpu', 'cuda']
        missing_fields = [field for field in required_fields if field not in result_lower]

        if missing_fields:
            print(f"[WARNING] Hardware analysis incomplete - missing: {', '.join(missing_fields)}")
            return False

        ram_match = re.search(r'ram_gb:\s*([\d.]+)', result_lower)
        if ram_match:
            print(f"[INFO] Detected {ram_match.group(1)} GB RAM")

        return True

    def _validate_reason(self, result: str, result_lower: str, step) -> bool:
        if len(result) < 200:
            print(f"[WARNING] Reasoning output seems too short ({len(result)} chars)")
            return False

        has_analysis = any(keyword in result_lower for keyword in
                           ['tradeoff', 'recommendation', 'strategy', 'approach'])

        if not has_analysis:
            print(f"[WARNING] Reasoning output lacks analysis keywords")
            return False

        return True

    def _validate_generic(self, result: str, result_lower: str, step) -> bool:
        error_indicators = [
            "error:",
            "failed to",
            "exception:",
            "traceback",
            "could not"
        ]

        for indicator in error_indicators:
            if indicator in result_lower:
                print(f"[ERROR] Tool output contains error indicator: '{indicator}'")
                return False

        return True

    def _check_dependencies(self, step, step_results: dict, failed_steps: set) -> bool:
        if step.tool == "code_analyze":
            for step_id in step_results:
                if step_results[step_id]['tool'] == 'ingest':
                    if not step_results[step_id].get('success', True):
                        print(f"[DEPENDENCY] code_analyze requires successful ingest (step {step_id} failed)")
                        return True

        if step.tool == "reason":
            analysis_tools = ['code_analyze', 'hardware_analyze']
            for step_id in step_results:
                if step_results[step_id]['tool'] in analysis_tools:
                    if not step_results[step_id].get('success', True):
                        print(f"[DEPENDENCY] reason requires successful analysis (step {step_id} failed)")
                        return True

        if step.tool == "code":
            for step_id in step_results:
                if step_results[step_id]['tool'] == 'reason':
                    if not step_results[step_id].get('success', True):
                        print(f"[DEPENDENCY] code generation requires successful reasoning (step {step_id} failed)")
                        return True

        return False

    def run(self, task: str, log_fn=None) -> str:
        """
        Run the full agent pipeline.

        Args:
            task: user query
            log_fn: optional callable(str) to receive log lines

        Returns:
            Combined log string (useful for UIs like Gradio)
        """
        logs = []

        def log(msg: str):
            msg = str(msg)
            logs.append(msg)
            print(msg)
            if log_fn is not None:
                log_fn(msg)

        log(f"[{self.name}] Task:\n{task}\n")

        plan = self.think(task, log_fn=log)

        # Optionally log a separator or summary if you like, but no pretty():
        log("\n[Plan ready]\n")

        step_results = self.execute(plan, log_fn=log)

        # Optionally, you could extract a "final answer" from step_results here.
        # For now we just return the concatenated logs.
        return "\n".join(logs)
