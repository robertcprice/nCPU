"""
GLM MCP Client Wrapper

Provides a simple interface to the GLM MCP server for LLM calls.
GLM is ~10x cheaper than OpenAI with comparable quality.
"""

from typing import Optional


def glm_ask(
    prompt: str,
    model: str = "sonnet",
    temperature: float = 0.3,
) -> str:
    """
    Ask GLM a question via MCP.

    Args:
        prompt: The question/prompt to send
        model: Model to use (haiku, sonnet, opus)
        temperature: Temperature (0-1)

    Returns:
        Model's response as string
    """
    try:
        from . import glm_agent as glm_mcp
        # Use the MCP tool
        result = glm_mcp.glm_ask(question=prompt, model=model)
        return result
    except ImportError:
        # Fallback - try direct MCP import
        try:
            import subprocess
            import json
            # This would require the MCP server to be running
            raise ImportError("GLM MCP server not configured")
        except Exception as e:
            return f"[GLM unavailable: {e}]"


def glm_summarize(
    text: str,
    style: str = "concise",
    model: str = "haiku",
) -> str:
    """
    Summarize text using GLM.

    Args:
        text: Text to summarize
        style: Summary style (concise, detailed, bullet-points, executive)
        model: Model to use

    Returns:
        Summary as string
    """
    try:
        import subprocess
        result = subprocess.run(
            ["glm-summarize", f"--style={style}", "--model", model],
            input=text,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except:
        return "[GLM summarize unavailable]"


def glm_analyze(
    task: str,
    working_directory: Optional[str] = None,
    model: str = "sonnet",
) -> str:
    """
    Analyze code using GLM with file access.

    Args:
        task: Analysis task
        working_directory: Directory to analyze
        model: Model to use

    Returns:
        Analysis results
    """
    try:
        import subprocess
        result = subprocess.run(
            ["glm-analyze", "--task", task, "--model", model],
            input=working_directory or "",
            capture_output=True,
            text=True,
        )
        return result.stdout
    except:
        return "[GLM analyze unavailable]"
