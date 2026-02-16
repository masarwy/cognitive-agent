def format_tool_input(data: dict) -> str:
    lines = []
    for k, v in data.items():
        if isinstance(v, str) and "\n" in v:
            lines.append(f"{k}: |")
            lines.extend(f"  {line}" for line in v.splitlines())
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)