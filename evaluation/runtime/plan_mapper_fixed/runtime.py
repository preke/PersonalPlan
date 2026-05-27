from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from types import MethodType
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from crewai import Agent
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from crewai_tools import (
        ArxivPaperTool,
        CodeDocsSearchTool,
        DirectoryReadTool,
        FileReadTool,
        FileWriterTool,
        FirecrawlSearchTool,
        ScrapeWebsiteTool,
        SerperDevTool,
        GithubSearchTool,
        RagTool,
    )
except Exception:  # noqa: BLE001
    ArxivPaperTool = None
    CodeDocsSearchTool = None
    DirectoryReadTool = None
    FileReadTool = None
    FileWriterTool = None
    FirecrawlSearchTool = None
    ScrapeWebsiteTool = None
    SerperDevTool = None
    GithubSearchTool = None
    RagTool = None

from .models import ExecutionReport, PlanPayload, RuntimeConfig, StepRunResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Docker availability check — runs once at import time
# ---------------------------------------------------------------------------
def _check_docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

USE_DOCKER: bool = _check_docker_available()
if not USE_DOCKER:
    logger.warning(
        "docker command not found or daemon not running — "
        "SafeCodeInterpreterTool will fall back to local execution. "
        "Install Docker and ensure the daemon is running to enable sandboxed execution."
    )

DOCKER_IMAGE = "mas-runtime:latest"


class _SafeCodeInput(BaseModel):
    code: str = Field(
        description=(
            "Code to execute. Supported: Python, C, C++, Java, JavaScript, TypeScript, "
            "Swift, R, Ruby, PHP, Kotlin, SQL/PostgreSQL."
        )
    )
    libraries_used: str = Field(
        default="",
        description=(
            "Language hint (e.g. 'java', 'c++', 'javascript', 'typescript', 'swift', "
            "'python', 'r', 'ruby', 'php', 'kotlin', 'sql', 'postgresql'). "
            "Used to pick the right runtime."
        ),
    )


def _detect_language(code: str, hint: str) -> str:
    """Return one of: python, c, cpp, java, javascript, typescript, swift, r, groovy,
    csharp, ruby, php, kotlin, sql."""
    # Normalise hint — LLM may pass a list coerced to string like "['JavaScript']"
    h = hint.lower() if isinstance(hint, str) else str(hint).lower()

    # Explicit hints checked before code heuristics
    if any(k in h for k in ("javascript", "js", "node")):
        return "javascript"
    if any(k in h for k in ("typescript", "ts")):
        return "typescript"
    if "swift" in h:
        return "swift"
    if "ruby" in h:
        return "ruby"
    if "groovy" in h:
        return "groovy"
    if any(k in h for k in ("csharp", "c#", ".net", "dotnet")):
        return "csharp"
    if "kotlin" in h:
        return "kotlin"
    if any(k in h for k in ("golang", "go")):
        return "go"
    if "rust" in h:
        return "rust"
    if "php" in h:
        return "php"
    if any(k in h for k in ("sql", "postgresql", "postgres", "sqlite", "mysql")):
        return "sql"
    if re.search(r"\br\b|rscript|tidyverse|ggplot|dplyr", h):
        return "r"

    # Code-based JS heuristics run BEFORE the "java" hint check because the
    # LLM sometimes passes libraries_used="java" for JavaScript code.
    if re.search(
        r"\bconsole\.log\b|\bPromise\b|\basync\s+function\b"
        r"|\bawait\b.*\bPromise\b|\brequire\(|\.then\(|=>",
        code,
    ):
        return "javascript"

    # Now honour remaining hints
    if any(k in h for k in ("java", "javac")):
        return "java"
    if any(k in h for k in ("c++", "cpp", "g++")):
        return "cpp"
    if "python" in h or "pip" in h:
        return "python"

    # Code-based heuristics for other languages
    if re.search(r":\s*(string|number|boolean|any)\b|interface\s+\w+\s*\{|\bReadonly<", code):
        return "typescript"
    if re.search(r"\bfunc\s+\w+\s*\(.*?\)\s*(->|\{)|\bvar\s+\w+\s*:\s*[A-Z]\w+\b", code):
        return "swift"
    if re.search(r"\bpublic\s+class\s+\w+", code):
        return "java"
    if re.search(r"#include\s*<(iostream|vector|string|map|algorithm|memory|functional)", code):
        return "cpp"
    if re.search(r"#include\s*<(stdio|stdlib|string|math|time)\.h>", code):
        return "c"
    if re.search(r"<-\s|\blibrary\(|\bdata\.frame\(|\bggplot\(|\bc\(", code):
        return "r"
    # SQL keyword heuristics (must come after other checks to avoid false positives)
    if re.search(
        r"\bSELECT\b|\bINSERT\s+INTO\b|\bCREATE\s+TABLE\b|\bUPDATE\b.*\bSET\b",
        code, re.IGNORECASE,
    ):
        return "sql"
    return "python"


def _run_subprocess(cmd: list, timeout: int = 30, cwd: str | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    out = result.stdout or ""
    if result.stderr:
        out += "\n[stderr]\n" + result.stderr[:500]
    return out.strip() or "[No output]"


class SafeCodeInterpreterTool(BaseTool):
    """Multi-language code interpreter. Runs inside a Docker container when available,
    falls back to direct subprocess execution otherwise."""

    name: str = "Code Interpreter"
    description: str = (
        "Executes code in an isolated environment. "
        "Supports Python, C, C++, Java, JavaScript, TypeScript, Swift, Ruby, R, PHP, Kotlin, Go, Rust, and SQL/PostgreSQL. "
        "All these languages ARE supported — do NOT simulate or describe output, always call this tool. "
        "Returns stdout/stderr. Times out after 60 seconds. "
        "Pass the language in 'libraries_used': 'python', 'java', 'c', 'c++', 'javascript', "
        "'typescript', 'swift', 'ruby', 'r', 'php', 'kotlin', 'sql', or 'postgresql'."
    )
    args_schema: type[BaseModel] = _SafeCodeInput

    def _run(self, code: str, libraries_used: str = "") -> str:
        lang = _detect_language(code, libraries_used)
        try:
            if lang == "python":
                return self._run_python(code)
            elif lang == "java":
                return self._run_java(code)
            elif lang == "cpp":
                return self._run_compiled(code, suffix=".cpp", compiler="g++")
            elif lang == "c":
                return self._run_compiled(code, suffix=".c", compiler="gcc")
            elif lang == "javascript":
                return self._run_js(code)
            elif lang == "typescript":
                return self._run_typescript(code)
            elif lang == "swift":
                return self._run_swift(code)
            elif lang == "ruby":
                return self._run_ruby(code)
            elif lang == "r":
                return self._run_r(code)
            elif lang == "php":
                return self._run_php(code)
            elif lang == "kotlin":
                return self._run_kotlin(code)
            elif lang == "sql":
                return self._run_sql(code)
            elif lang == "go":
                return self._run_go(code)
            elif lang == "rust":
                return self._run_rust(code)
            elif lang == "groovy":
                return "[Groovy runtime not installed. Cannot execute Groovy code.]"
            elif lang == "csharp":
                return self._run_csharp(code)
            else:
                return self._run_python(code)
        except subprocess.TimeoutExpired:
            return "[Code execution timed out after 60s]"
        except FileNotFoundError as exc:
            return f"[Runtime not found: {exc}. Is the compiler/interpreter installed?]"
        except Exception as exc:  # noqa: BLE001
            return f"[Execution error: {exc}]"

    # ── Docker helper ────────────────────────────────────────────────────────

    def _run_in_docker(
        self,
        code: str,
        suffix: str,
        cmd: list,
        timeout: int = 60,
    ) -> str:
        """Write *code* to a temp file, mount it into the container, and run *cmd*.

        cmd should reference the file as ``/workspace/main{suffix}``.
        """
        with tempfile.NamedTemporaryFile(
            suffix=suffix, mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp = f.name
        try:
            docker_path = f"/workspace/main{suffix}"
            docker_cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "-v", f"{tmp}:{docker_path}:ro",
                DOCKER_IMAGE,
            ] + cmd + [docker_path]
            result = subprocess.run(
                docker_cmd,
                capture_output=True, text=True, timeout=timeout,
            )
            out = result.stdout or ""
            if result.stderr:
                out += "\n[stderr]\n" + result.stderr[:500]
            return out.strip() or "[No output]"
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run_in_docker_dir(
        self,
        files: dict[str, str],
        cmd: list,
        timeout: int = 60,
        network: bool = False,
    ) -> str:
        """Mount an entire temp directory into /workspace and run *cmd* inside it.

        *files* is {filename: content}. cmd must be a complete command list
        (no auto-appended path).
        """
        tmpdir = tempfile.mkdtemp()
        try:
            for name, content in files.items():
                path = os.path.join(tmpdir, name)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            network_flag = [] if network else ["--network", "none"]
            docker_cmd = [
                "docker", "run", "--rm",
                *network_flag,
                "-v", f"{tmpdir}:/workspace",
                "-w", "/workspace",
                DOCKER_IMAGE,
            ] + cmd
            result = subprocess.run(
                docker_cmd,
                capture_output=True, text=True, timeout=timeout,
            )
            out = result.stdout or ""
            if result.stderr:
                out += "\n[stderr]\n" + result.stderr[:500]
            return out.strip() or "[No output]"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Language runners ─────────────────────────────────────────────────────

    def _run_python(self, code: str) -> str:
        if USE_DOCKER:
            return self._run_in_docker(code, ".py", ["python3"])
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            return _run_subprocess([sys.executable, tmp])
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run_compiled(self, code: str, suffix: str, compiler: str) -> str:
        if USE_DOCKER:
            # Compile then run inside one container invocation via shell
            shell_cmd = f"{compiler} /workspace/main{suffix} -o /workspace/main && /workspace/main"
            return self._run_in_docker(code, suffix, ["sh", "-c", shell_cmd.replace("/workspace/main" + suffix, "$1").replace("$1", "/workspace/main" + suffix)])
        # Fallback: local compile
        tmpdir = tempfile.mkdtemp()
        src = os.path.join(tmpdir, "main" + suffix)
        exe = os.path.join(tmpdir, "main")
        try:
            with open(src, "w", encoding="utf-8") as f:
                f.write(code)
            compile_result = subprocess.run(
                [compiler, src, "-o", exe],
                capture_output=True, text=True, timeout=30,
            )
            if compile_result.returncode != 0:
                return f"[Compile error — {compiler} toolchain available but code has errors]\n{compile_result.stderr[:500]}"
            return _run_subprocess([exe], cwd=tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _run_compiled_docker(self, code: str, suffix: str, compiler: str) -> str:
        """Compile + run inside Docker using a directory mount so both compile and run happen in the same dir."""
        filename = "main" + suffix
        compile_and_run = f"{compiler} /workspace/{filename} -o /workspace/main && /workspace/main"
        return self._run_in_docker_dir(
            {filename: code},
            ["sh", "-c", compile_and_run],
        )

    def _run_java(self, code: str) -> str:
        m = re.search(r"\bpublic\s+class\s+(\w+)", code)
        classname = m.group(1) if m else "Main"
        if USE_DOCKER:
            filename = classname + ".java"
            compile_and_run = f"javac /workspace/{filename} && java -cp /workspace {classname}"
            return self._run_in_docker_dir(
                {filename: code},
                ["sh", "-c", compile_and_run],
                timeout=60,
            )
        tmpdir = tempfile.mkdtemp()
        src = os.path.join(tmpdir, classname + ".java")
        try:
            with open(src, "w", encoding="utf-8") as f:
                f.write(code)
            compile_result = subprocess.run(
                ["javac", src],
                capture_output=True, text=True, timeout=30, cwd=tmpdir,
            )
            if compile_result.returncode != 0:
                return f"[Compile error — javac available but code has errors]\n{compile_result.stderr[:500]}"
            return _run_subprocess(["java", classname], timeout=30, cwd=tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _run_js(self, code: str) -> str:
        if USE_DOCKER:
            return self._run_in_docker(code, ".js", ["node"])
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            return _run_subprocess(["node", tmp])
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run_r(self, code: str) -> str:
        if USE_DOCKER:
            return self._run_in_docker(code, ".R", ["Rscript"])
        with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            return _run_subprocess(["Rscript", tmp])
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run_typescript(self, code: str) -> str:
        if USE_DOCKER:
            return self._run_in_docker(code, ".ts", ["npx", "tsx"], timeout=60)
        with tempfile.NamedTemporaryFile(suffix=".ts", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            return _run_subprocess(["npx", "tsx", tmp], timeout=60)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run_ruby(self, code: str) -> str:
        if USE_DOCKER:
            return self._run_in_docker(code, ".rb", ["ruby"])
        with tempfile.NamedTemporaryFile(suffix=".rb", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            return _run_subprocess(["ruby", tmp], timeout=60)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run_swift(self, code: str) -> str:
        if USE_DOCKER:
            # Swift is NOT in the Docker image — fall back to local if available
            try:
                return self._run_in_docker(code, ".swift", ["swift"])
            except Exception:
                pass
        with tempfile.NamedTemporaryFile(suffix=".swift", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            return _run_subprocess(["swift", tmp], timeout=60)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run_php(self, code: str) -> str:
        if USE_DOCKER:
            return self._run_in_docker(code, ".php", ["php"])
        with tempfile.NamedTemporaryFile(suffix=".php", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            return _run_subprocess(["php", tmp], timeout=60)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run_kotlin(self, code: str) -> str:
        """Compile with kotlinc and run the resulting jar."""
        if USE_DOCKER:
            # kotlinc outputs a jar; we compile and run in one shell command
            compile_and_run = (
                "kotlinc /workspace/main.kt -include-runtime -d /workspace/main.jar 2>&1 "
                "&& kotlin -classpath /workspace/main.jar MainKt"
            )
            return self._run_in_docker_dir(
                {"main.kt": code},
                ["sh", "-c", compile_and_run],
                timeout=120,
            )
        tmpdir = tempfile.mkdtemp()
        src = os.path.join(tmpdir, "main.kt")
        jar = os.path.join(tmpdir, "main.jar")
        try:
            with open(src, "w", encoding="utf-8") as f:
                f.write(code)
            compile_result = subprocess.run(
                ["kotlinc", src, "-include-runtime", "-d", jar],
                capture_output=True, text=True, timeout=90,
            )
            if compile_result.returncode != 0:
                return f"[Kotlin compile error]\n{compile_result.stderr[:500]}"
            return _run_subprocess(["kotlin", "-classpath", jar, "MainKt"], timeout=60, cwd=tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _run_csharp(self, code: str) -> str:
        """Run C# via dotnet-script or a temporary csproj."""
        if USE_DOCKER:
            # Write a minimal Program.cs and run with dotnet-script / csc approach.
            # We use `dotnet run` with a minimal project file.
            csproj = (
                "<Project Sdk=\"Microsoft.NET.Sdk\">\n"
                "  <PropertyGroup>\n"
                "    <OutputType>Exe</OutputType>\n"
                "    <TargetFramework>net8.0</TargetFramework>\n"
                "    <Nullable>enable</Nullable>\n"
                "  </PropertyGroup>\n"
                "</Project>\n"
            )
            compile_and_run = "dotnet run --project /workspace/app.csproj"
            return self._run_in_docker_dir(
                {"Program.cs": code, "app.csproj": csproj},
                ["sh", "-c", compile_and_run],
                timeout=120,
            )
        with tempfile.NamedTemporaryFile(suffix=".cs", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            return _run_subprocess(["dotnet-script", tmp], timeout=90)
        except FileNotFoundError:
            return "[C# (.NET) runtime not installed on this machine. Cannot execute C# code.]"
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _run_sql(self, code: str) -> str:
        """Execute SQL via psql inside Docker (PostgreSQL teaching DB) or psql locally."""
        if USE_DOCKER:
            # PostgreSQL is running inside the container managed by /entrypoint.sh.
            # We need network access to the container's postgres — easier to run psql
            # directly inside a container that also has postgres running.
            # Strategy: write SQL to a file, pass it to psql via stdin in the container.
            # We start postgres inside the container with a one-shot entrypoint override.
            sql_file = "query.sql"
            run_cmd = (
                "service postgresql start && "
                "until pg_isready -q; do sleep 0.1; done && "
                "psql -U mas -d teaching -f /workspace/query.sql"
            )
            return self._run_in_docker_dir(
                {sql_file: code},
                ["sh", "-c", run_cmd],
                timeout=60,
                network=True,  # postgres loopback needs network
            )
        # Fallback: local psql
        with tempfile.NamedTemporaryFile(suffix=".sql", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            return _run_subprocess(
                ["psql", "-U", "mas", "-d", "teaching", "-f", tmp],
                timeout=60,
            )
        except FileNotFoundError:
            return "[psql not found. Install PostgreSQL client or use Docker mode.]"
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass


    def _run_go(self, code: str) -> str:
        if USE_DOCKER:
            return self._run_in_docker_dir(
                {"main.go": code},
                ["sh", "-c", "cd /workspace && go run main.go"],
                timeout=60,
            )
        with tempfile.NamedTemporaryFile(suffix=".go", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code); tmp = f.name
        try:
            return _run_subprocess(["go", "run", tmp], timeout=60)
        except FileNotFoundError:
            return "[Go runtime not found. Install Go or use Docker mode.]"
        finally:
            try: os.unlink(tmp)
            except OSError: pass

    def _run_rust(self, code: str) -> str:
        if USE_DOCKER:
            return self._run_in_docker_dir(
                {"main.rs": code},
                ["sh", "-c", "cd /workspace && rustc main.rs -o main && ./main"],
                timeout=90,
            )
        tmpdir = tempfile.mkdtemp()
        src = os.path.join(tmpdir, "main.rs")
        exe = os.path.join(tmpdir, "main")
        try:
            with open(src, "w", encoding="utf-8") as f:
                f.write(code)
            compile_result = subprocess.run(
                ["rustc", src, "-o", exe],
                capture_output=True, text=True, timeout=60,
            )
            if compile_result.returncode != 0:
                return f"[Compile error — rustc IS available but your code has errors]\n{compile_result.stderr[:500]}"
            return _run_subprocess([exe], timeout=30, cwd=tmpdir)
        except FileNotFoundError:
            return "[Rust compiler not found. Install Rust or use Docker mode.]"
        finally:
            import shutil; shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Overrides for compiled languages to use the directory-mount approach in Docker
# (gcc/g++ need to write an intermediate binary — the base _run_compiled doesn't
#  handle that cleanly; redirect to _run_compiled_docker when USE_DOCKER is True)
# ---------------------------------------------------------------------------
_orig_run_compiled = SafeCodeInterpreterTool._run_compiled

def _run_compiled_patched(self, code: str, suffix: str, compiler: str) -> str:
    if USE_DOCKER:
        return self._run_compiled_docker(code, suffix, compiler)
    return _orig_run_compiled(self, code, suffix, compiler)

SafeCodeInterpreterTool._run_compiled = _run_compiled_patched  # type: ignore[method-assign]


TOOL_ARG_HINTS: Dict[str, str] = {
    "CodeDocsSearchTool": "When calling this tool, pass EXACT keys: {'search_query': <string>, 'docs_url': <string>}. Do not use 'description' as a key.",
    "CodeInterpreterTool": (
        "When calling this tool, pass EXACT keys: {'code': <string>, 'libraries_used': <string>}. "
        "Set libraries_used to the language: 'python', 'java', 'c', 'c++', 'javascript', "
        "'typescript', 'swift', 'ruby', 'r', 'php', 'kotlin', 'sql', or 'postgresql'. "
        "ALWAYS call this tool with real code — never simulate or describe the output."
    ),
    "GithubSearchTool": "When calling this tool, pass EXACT keys: {'search_query': <string>, 'github_repo': <string>, 'content_types': <list>}.",
}


# Code-execution encouragement appended to prompts when step.tool == CodeInterpreterTool.
# Documented in PROMPT_CHANGELOG.md (2026-05-22). To revert: delete this constant and
# the two call sites in _run_interactive_step / _run_step.
CODE_TOOL_NUDGE = (
    "Code-execution guidance (this tool compiles and runs code in a sandbox; "
    "treat its output as the source of truth):\n"
    "- For compiled languages (java, c, c++, kotlin, swift, typescript), you MUST "
    "compile via this tool before claiming the code is correct; quote real compile "
    "errors verbatim if they appear.\n"
    "- Quote the real stdout/stderr returned by the tool. Do not write phrases like "
    "\"Expected output:\", \"you would get:\", \"this would produce\", or "
    "\"the result would be\" — those signal you did not actually execute.\n"
    "- Pedagogical ordering: only run/compile AFTER the learner has predicted or "
    "attempted. The tool belongs in validation / feedback / consolidation, not in "
    "the first probe.\n"
    "- Before sending your message, self-check: did I actually run or compile this "
    "code, or am I describing what it would do?\n"
)


class _CodeDocsSearchCompatInput(BaseModel):
    search_query: str = Field(default="", description="Search query for docs lookup")
    docs_url: str = Field(description="Target docs URL")
    description: Optional[str] = Field(default=None, description="Alias for search_query")


def _make_codedocs_tool_compat() -> Any:
    if CodeDocsSearchTool is None:
        raise RuntimeError("CodeDocsSearchTool is unavailable")
    tool = CodeDocsSearchTool()
    tool.args_schema = _CodeDocsSearchCompatInput
    original_run = tool._run

    def _compat_run(
        self: Any,
        search_query: str = "",
        docs_url: str = "",
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        query = search_query or description or kwargs.get("query", "")
        if not docs_url:
            return f"[CodeDocsSearchTool] No docs_url provided for query: {query!r}. Please specify a valid documentation URL."
        try:
            return original_run(search_query=query, docs_url=docs_url)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            return f"[CodeDocsSearchTool] Failed to fetch {docs_url!r}: {msg}. The agent should describe the concept from its training knowledge instead."

    tool._run = MethodType(_compat_run, tool)
    return tool


class ToolRunner:
    def run(self, tool_name: str, step_id: str, instruction: str, mode: str) -> StepRunResult:
        if mode == "smoke":
            return StepRunResult(
                step_id=step_id,
                status="PASS",
                content=f"[SMOKE] Tool {tool_name} executed for {step_id}",
                meta={"tool": tool_name},
            )

        if tool_name == "CodeInterpreterTool":
            return StepRunResult(
                step_id=step_id,
                status="PASS",
                content="CodeInterpreterTool execution finished.",
                meta={"tool": tool_name},
            )

        if tool_name == "CodeDocsSearchTool":
            return StepRunResult(
                step_id=step_id,
                status="PASS",
                content="CodeDocsSearchTool lookup finished.",
                meta={"tool": tool_name},
            )

        return StepRunResult(
            step_id=step_id,
            status="FAIL",
            content=f"Unsupported tool: {tool_name}",
            meta={"tool": tool_name},
        )


class PlanRuntime:
    def __init__(self, plan: PlanPayload, config: RuntimeConfig, run_dir: Path):
        self.plan = plan
        self.config = config
        self.run_dir = run_dir
        self.events: List[str] = []
        self.outputs: Dict[str, StepRunResult] = {}
        self.step_defs = self._build_step_index()
        self.step_to_agent = self._build_step_agent_index()
        self.agent_tool_names: Dict[str, set[str]] = {}
        self.agents = self._build_agents()
        self.tool_runner = ToolRunner()
        self.loop_eval_counts: Dict[str, int] = {}
        self.loop_condition_step_ids: set[str] = self._find_loop_condition_steps()
        self.student_agent = self._build_student_agent() if self.config.mode == "live" else None
        self.execution_log: List[Dict[str, Any]] = []
        self._current_loop_context: Optional[Dict[str, Any]] = None

    def _sanitize_text(self, text: str) -> str:
        cleaned = []
        for ch in text:
            o = ord(ch)
            if ch in {"\n", "\t"} or 32 <= o <= 126:
                cleaned.append(ch)
            else:
                cleaned.append(" ")
        return "".join(cleaned)

    def _build_agents(self) -> Dict[str, Agent]:
        agents: Dict[str, Agent] = {}
        for a in self.plan.output.agents:
            tools, tool_names = self._resolve_tools(a.tools)
            self.agent_tool_names[a.agent_role] = set(tool_names)
            agents[a.agent_role] = Agent(
                role=a.agent_role,
                goal=a.goal,
                backstory=a.description,
                tools=tools,
                verbose=False,
                allow_delegation=False,
                llm=self.config.model,
            )
        return agents

    def _resolve_tools(self, tool_names: List[str]) -> tuple[List[Any], List[str]]:
        resolved: List[Any] = []
        resolved_names: List[str] = []
        for name in tool_names:
            try:
                if name == "FirecrawlSearchTool" and FirecrawlSearchTool is not None:
                    if not os.getenv("FIRECRAWL_API_KEY"):
                        self._log("TOOL FirecrawlSearchTool skipped: missing FIRECRAWL_API_KEY")
                    else:
                        resolved.append(FirecrawlSearchTool())
                        resolved_names.append(name)
                elif name == "RagTool" and RagTool is not None:
                    resolved.append(RagTool())
                    resolved_names.append(name)
                elif name == "CodeInterpreterTool":
                    resolved.append(SafeCodeInterpreterTool())
                    resolved_names.append(name)
                elif name == "DirectoryReadTool" and DirectoryReadTool is not None:
                    resolved.append(DirectoryReadTool())
                    resolved_names.append(name)
                elif name == "FileReadTool" and FileReadTool is not None:
                    resolved.append(FileReadTool())
                    resolved_names.append(name)
                elif name == "FileWriterTool" and FileWriterTool is not None:
                    resolved.append(FileWriterTool())
                    resolved_names.append(name)
                elif name == "GithubSearchTool" and GithubSearchTool is not None:
                    gh_token = os.getenv("GITHUB_TOKEN")
                    if gh_token:
                        resolved.append(GithubSearchTool(gh_token=gh_token))
                        resolved_names.append(name)
                    else:
                        self._log(
                            "TOOL GithubSearchTool skipped: missing GITHUB_TOKEN environment variable"
                        )
                elif name == "CodeDocsSearchTool" and CodeDocsSearchTool is not None:
                    resolved.append(_make_codedocs_tool_compat())
                    resolved_names.append(name)
                elif name == "ArxivPaperTool" and ArxivPaperTool is not None:
                    resolved.append(ArxivPaperTool())
                    resolved_names.append(name)
                elif name == "SerperDevTool" and SerperDevTool is not None:
                    if not os.getenv("SERPER_API_KEY"):
                        self._log("TOOL SerperDevTool skipped: missing SERPER_API_KEY")
                    else:
                        resolved.append(SerperDevTool())
                        resolved_names.append(name)
                elif name == "ScrapeWebsiteTool" and ScrapeWebsiteTool is not None:
                    resolved.append(ScrapeWebsiteTool())
                    resolved_names.append(name)
                else:
                    self._log(f"TOOL {name} unsupported or unavailable in current environment")
            except Exception as exc:  # noqa: BLE001
                self._log(f"TOOL {name} init failed: {exc}")
        return resolved, resolved_names

    def _build_step_index(self) -> Dict[str, Dict[str, Any]]:
        steps: Dict[str, Dict[str, Any]] = {}
        for subtask in self.plan.output.subtasks:
            for step in subtask.steps:
                steps[step.id] = {
                    "subtask_id": subtask.id,
                    "subtask_name": subtask.name,
                    "agent": step.agent,
                    "step": step,
                }
        return steps

    def _build_step_agent_index(self) -> Dict[str, str]:
        return {sid: data["agent"] for sid, data in self.step_defs.items()}

    def _find_loop_condition_steps(self) -> set[str]:
        """Extract step IDs referenced in loop conditions for generic smoke behavior."""
        cond_pattern = re.compile(r"^([A-Za-z0-9\-]+)\.")
        step_ids: set[str] = set()
        for item in self.plan.output.execution_order:
            if not isinstance(item, str) and hasattr(item, "loop"):
                m = cond_pattern.match(item.loop.condition.strip())
                if m:
                    step_ids.add(m.group(1))
        return step_ids

    def _log(self, message: str) -> None:
        timestamp = datetime.now(UTC).isoformat()
        line = f"{timestamp} {message}"
        self.events.append(line)
        print(line, flush=True)

    def _dependencies_met(self, step_id: str) -> bool:
        step = self.step_defs[step_id]["step"]
        return all(dep in self.outputs for dep in step.depends_on)

    def _evaluate_condition(self, expression: str) -> bool:
        # Supports forms like:
        #   S5-2.output == 'FAIL'
        #   S3-4.implementation_correct == false
        m = re.match(r"^([A-Za-z0-9\-]+)\.([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(.+)$", expression.strip())
        if not m:
            return False
        step_id, field, rhs_raw = m.group(1), m.group(2), m.group(3).strip()

        result = self.outputs.get(step_id)
        if not result:
            return False

        # Parse the right-hand side value
        rhs_raw_lower = rhs_raw.lower()
        if rhs_raw_lower == "true":
            rhs: Any = True
        elif rhs_raw_lower == "false":
            rhs = False
        elif (rhs_raw.startswith("'") and rhs_raw.endswith("'")) or (rhs_raw.startswith('"') and rhs_raw.endswith('"')):
            rhs = rhs_raw[1:-1]
        else:
            rhs = rhs_raw

        # For the legacy "output" field, compare against status
        if field == "output":
            return result.status == rhs

        # For other fields, try to extract from the step's content (as JSON or text)
        content = result.content or ""
        # Try JSON extraction
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and field in obj:
                val = obj[field]
                if isinstance(val, bool) and isinstance(rhs, bool):
                    return val == rhs
                return str(val).lower() == str(rhs).lower()
        except Exception:
            pass

        # Try regex extraction from text
        pat = re.compile(rf"{re.escape(field)}\s*[:=]\s*(true|false|\S+)", re.IGNORECASE)
        fm = pat.search(content)
        if fm:
            found = fm.group(1).strip()
            return found.lower() == str(rhs).lower()

        # Field not found in output — conservatively assume condition still holds
        # (avoids incorrectly skipping remediation when code execution failed)
        return True

    def _extract_topic_blindness(self) -> str:
        """Extract key concepts from the plan that the student should NOT know."""
        concepts = []
        for subtask in self.plan.output.subtasks:
            obj = subtask.subtask_objective
            if obj:
                concepts.append(obj)
        # Deduplicate and format
        if not concepts:
            return ""
        lines = ["TOPIC BLINDNESS — You have NEVER encountered these concepts:"]
        for i, c in enumerate(concepts, 1):
            lines.append(f"   {i}. {c}")
        lines.append(
            "You MUST NOT state, use, or reference ANY of the above ideas\n"
            "   until the instructor explicitly teaches them to you in THIS conversation.\n"
            "   If asked to predict, you should guess INCORRECTLY or say you don't know."
        )
        return "\n".join(lines)

    def _build_student_agent(self) -> Agent:
        """Create a simulated student agent from the learner profile."""
        learner = self.plan.input.learner
        skills = ", ".join(learner.skills) if learner.skills else "no listed skills"
        topic_blindness = self._extract_topic_blindness()
        backstory = (
            "You are a learner interacting with a teaching system.\n"
            "You are here because you encountered a programming problem\n"
            "you cannot solve on your own.\n\n"
            "Your background:\n"
            f"- Description: {learner.self_description}\n"
            f"- Skills: {skills}\n\n"
            "The problem you need help with:\n"
            f"{self.plan.input.query}\n\n"
            "You found this question because you could not solve it.\n"
            "Respond accordingly.\n\n"
            "Behavioral rules:\n\n"
            "1. CORE CONSTRAINT\n"
            "   You do NOT know the answer to the problem being taught.\n"
            "   This is non-negotiable. You are someone who needs to learn\n"
            "   this topic through this conversation.\n\n"
            f"   {topic_blindness}\n\n"
            "2. REASONING FROM BACKGROUND\n"
            "   You may use your declared skills as reference frames.\n"
            "   For example, if you know Python and the topic is Java,\n"
            '   you may say "In Python I would do X, is Java similar?"\n'
            "   But your reasoning about the unfamiliar target topic\n"
            '   should be tentative - use phrases like "I think maybe",\n'
            '   "my guess would be", "I\'m not sure but".\n\n'
            "3. LEARNING PROGRESSION\n"
            "   - Before any instruction: uncertain, may guess WRONG\n"
            "   - After initial explanation: partial understanding, may still have gaps\n"
            "   - After targeted feedback: clearer grasp, can apply to the specific case\n"
            "   - After practice with correction: confident and correct\n\n"
            "4. AUTHENTICITY\n"
            "   - If confused, say so.\n"
            "   - If an explanation helped, say what specifically helped and why.\n"
            "   - If asked to write code, your first attempt SHOULD have bugs.\n"
            "   - Do NOT produce perfect answers before receiving sufficient instruction.\n"
            "   - Do NOT reference information not yet presented in this conversation."
        )
        return Agent(
            role="Student Learner",
            goal=f"Learn and understand: {self.plan.input.query[:150]}",
            backstory=backstory,
            tools=[],
            allow_delegation=False,
            verbose=False,
            llm=self.config.student_model or self.config.model,
        )

    def _kickoff_with_retry(self, agent: Any, prompt: str, max_retries: int = 3) -> str:
        import time
        last_exc: Exception = RuntimeError("unknown")
        for attempt in range(max_retries):
            try:
                result = agent.kickoff(prompt)
                return result.raw
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if any(k in msg for k in ("connection", "timeout", "rate limit", "503", "502", "429")):
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    self._log(f"API error (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {exc}")
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    def _build_log_entry(self, step_id: str) -> Dict[str, Any]:
        """Build a structured execution log entry per spec Section 3.2."""
        data = self.step_defs[step_id]
        step = data["step"]
        loop_ctx = dict(self._current_loop_context) if self._current_loop_context else {
            "in_loop": False,
            "iteration": None,
            "exit_reason": None,
        }
        return {
            "step_id": step_id,
            "subtask_id": data["subtask_id"],
            "agent_role": data["agent"],
            "requires_human_input": step.requires_human_input,
            "plan_instruction": step.instruction,
            "plan_expected_output": step.expected_output,
            "actual_interaction": {},
            "loop_context": loop_ctx,
        }

    def _run_interactive_step(self, step_id: str) -> StepRunResult:
        """Run a step that requires human input using teacher-student simulation."""
        data = self.step_defs[step_id]
        step = data["step"]
        started_at = datetime.now(UTC).isoformat()

        agent = self.agents[data["agent"]]

        # Build context from ALL previous outputs
        context_parts = []
        for prev_sid, prev_result in self.outputs.items():
            if prev_sid != step_id:
                role = self.step_defs[prev_sid]["agent"]
                context_parts.append(f"[{role}] (Step {prev_sid}): {prev_result.content[:300]}")
        context_text = "\n".join(context_parts[-20:])

        # Run teacher agent
        teacher_prompt = (
            f"Objective: {step.objective}\n"
            f"Instruction: {step.instruction}\n"
            f"Expected Output: {step.expected_output}\n"
            f"Context:\n{context_text}\n"
            + ((f"Required Tool: {step.tool}. You MUST invoke this tool to compile/run the code in the sandbox; do not describe expected output.\n"
                if step.tool == "CodeInterpreterTool"
                else f"Required Tool: {step.tool}. Use this tool when needed.\n")
               if step.tool else "")
            + (f"Tool Input Contract: {TOOL_ARG_HINTS.get(step.tool, '')}\n" if step.tool else "")
            + (CODE_TOOL_NUDGE if step.tool == "CodeInterpreterTool" else "")
            + "Provide your instructional message to the student."
        )
        try:
            teacher_output = self._kickoff_with_retry(agent, teacher_prompt)
        except Exception as exc:
            teacher_output = f"[Teacher error: {exc}]"

        # Run student agent to respond
        # Student only sees teacher's current output, NOT prior step context
        # This prevents the student from seeing prior step answers and "cheating"
        student_prompt = (
            f"The instructor has sent you the following message. "
            f"Respond as a learner:\n\n"
            f"{teacher_output}"
        )
        try:
            student_output = self._kickoff_with_retry(self.student_agent, student_prompt)
        except Exception as exc:
            student_output = f"[Student error: {exc}]"

        # Store student_output as the step result (expected_output describes what learner produces)
        res = StepRunResult(
            step_id=step_id,
            status="PASS",
            content=student_output,
            meta={
                "agent": data["agent"],
                "tool": step.tool,
                "objective": step.objective,
                "instruction": step.instruction,
                "expected_output": step.expected_output,
                "depends_on": step.depends_on,
                "subtask_id": data["subtask_id"],
                "subtask_name": data["subtask_name"],
                "started_at": started_at,
                "ended_at": datetime.now(UTC).isoformat(),
                "teacher_output": teacher_output,
                "interactive": True,
            },
        )
        self.outputs[step_id] = res
        self._log(f"STEP {step_id} {res.status} interactive (teacher+student)")

        # Record structured execution log entry
        log_entry = self._build_log_entry(step_id)
        log_entry["actual_interaction"] = {
            "teacher_output": teacher_output,
            "student_response": student_output,
        }
        self.execution_log.append(log_entry)

        return res

    def _run_step(self, step_id: str) -> StepRunResult:
        data = self.step_defs[step_id]
        step = data["step"]
        started_at = datetime.now(UTC).isoformat()
        self._log(
            f"STEP {step_id} START agent={data['agent']} tool={step.tool or 'none'} objective={step.objective}"
        )
        if not self._dependencies_met(step_id):
            missing = [d for d in step.depends_on if d not in self.outputs]
            res = StepRunResult(
                step_id=step_id,
                status="ERROR",
                content=f"Dependencies not met: {missing}",
                meta={
                    "missing_dependencies": missing,
                    "objective": step.objective,
                    "instruction": step.instruction,
                    "expected_output": step.expected_output,
                    "depends_on": step.depends_on,
                    "subtask_id": data["subtask_id"],
                    "subtask_name": data["subtask_name"],
                    "agent": data["agent"],
                    "started_at": started_at,
                    "ended_at": datetime.now(UTC).isoformat(),
                },
            )
            self.outputs[step_id] = res
            self._log(f"STEP {step_id} ERROR dependencies missing")
            log_entry = self._build_log_entry(step_id)
            log_entry["actual_interaction"] = {"agent_output": res.content}
            self.execution_log.append(log_entry)
            return res

        if self.config.mode == "live" and step.tool:
            available = self.agent_tool_names.get(data["agent"], set())
            if step.tool not in available:
                res = StepRunResult(
                    step_id=step_id,
                    status="ERROR",
                    content=(
                        f"Required tool {step.tool} is not available for agent {data['agent']}. "
                        "Check tool installation and environment variables."
                    ),
                    meta={
                        "agent": data["agent"],
                        "required_tool": step.tool,
                        "available_tools": sorted(list(available)),
                        "objective": step.objective,
                        "instruction": step.instruction,
                        "expected_output": step.expected_output,
                        "depends_on": step.depends_on,
                        "subtask_id": data["subtask_id"],
                        "subtask_name": data["subtask_name"],
                        "started_at": started_at,
                        "ended_at": datetime.now(UTC).isoformat(),
                    },
                )
                self.outputs[step_id] = res
                self._log(f"STEP {step_id} ERROR missing required tool {step.tool}")
                log_entry = self._build_log_entry(step_id)
                log_entry["actual_interaction"] = {"agent_output": res.content}
                self.execution_log.append(log_entry)
                return res

        if step.tool and self.config.mode == "smoke":
            res = self.tool_runner.run(step.tool, step_id, step.instruction, self.config.mode)
            self.outputs[step_id] = res
            self._log(f"STEP {step_id} {res.status} via tool {step.tool}")
            log_entry = self._build_log_entry(step_id)
            log_entry["actual_interaction"] = {"agent_output": res.content}
            self.execution_log.append(log_entry)
            return res

        if self.config.mode == "smoke":
            # For steps referenced in loop conditions, alternate FAIL/PASS
            # so that loops exercise at least one retry before passing.
            if step_id in self.loop_condition_step_ids:
                count = self.loop_eval_counts.get(step_id, 0)
                self.loop_eval_counts[step_id] = count + 1
                status = "FAIL" if count % 2 == 0 else "PASS"
            else:
                status = "PASS"

            res = StepRunResult(
                step_id=step_id,
                status=status,
                content=f"[SMOKE] Executed {step_id}: {step.objective}",
                meta={
                    "agent": data["agent"],
                    "objective": step.objective,
                    "instruction": step.instruction,
                    "expected_output": step.expected_output,
                    "depends_on": step.depends_on,
                    "subtask_id": data["subtask_id"],
                    "subtask_name": data["subtask_name"],
                    "started_at": started_at,
                    "ended_at": datetime.now(UTC).isoformat(),
                },
            )
            self.outputs[step_id] = res
            self._log(f"STEP {step_id} {res.status} smoke")
            log_entry = self._build_log_entry(step_id)
            log_entry["actual_interaction"] = {"agent_output": res.content}
            self.execution_log.append(log_entry)
            return res

        # For interactive steps in live mode, use teacher-student simulation
        if step.requires_human_input and self.config.mode == "live" and self.student_agent is not None:
            return self._run_interactive_step(step_id)

        agent = self.agents[data["agent"]]
        # Include all previous outputs as context, not just depends_on
        context_parts = []
        for prev_sid, prev_result in self.outputs.items():
            if prev_sid != step_id:
                role = self.step_defs[prev_sid]["agent"]
                context_parts.append(f"[{role}] (Step {prev_sid}): {prev_result.content[:300]}")
        context_text = "\n".join(context_parts[-20:])  # Last 20 entries
        prompt = (
            f"Objective: {step.objective}\n"
            f"Instruction: {step.instruction}\n"
            f"Expected Output: {step.expected_output}\n"
            f"Context:\n{context_text}\n"
            + ((f"Required Tool: {step.tool}. You MUST invoke this tool to compile/run the code in the sandbox; do not describe expected output.\n"
                if step.tool == "CodeInterpreterTool"
                else f"Required Tool: {step.tool}. Use this tool when needed.\n")
               if step.tool else "")
            + (f"Tool Input Contract: {TOOL_ARG_HINTS.get(step.tool, '')}\n" if step.tool else "")
            + (CODE_TOOL_NUDGE if step.tool == "CodeInterpreterTool" else "")
            + "Return a concise execution response."
        )
        try:
            output_raw = self._kickoff_with_retry(agent, prompt)
            res = StepRunResult(
                step_id=step_id,
                status="PASS",
                content=output_raw,
                meta={
                    "agent": data["agent"],
                    "tool": step.tool,
                    "objective": step.objective,
                    "instruction": step.instruction,
                    "expected_output": step.expected_output,
                    "depends_on": step.depends_on,
                    "subtask_id": data["subtask_id"],
                    "subtask_name": data["subtask_name"],
                    "started_at": started_at,
                    "ended_at": datetime.now(UTC).isoformat(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            res = StepRunResult(
                step_id=step_id,
                status="ERROR",
                content=f"Live execution error: {exc}",
                meta={
                    "agent": data["agent"],
                    "tool": step.tool,
                    "objective": step.objective,
                    "instruction": step.instruction,
                    "expected_output": step.expected_output,
                    "depends_on": step.depends_on,
                    "subtask_id": data["subtask_id"],
                    "subtask_name": data["subtask_name"],
                    "started_at": started_at,
                    "ended_at": datetime.now(UTC).isoformat(),
                },
            )

        self.outputs[step_id] = res
        self._log(f"STEP {step_id} {res.status} live")
        log_entry = self._build_log_entry(step_id)
        log_entry["actual_interaction"] = {"agent_output": res.content}
        self.execution_log.append(log_entry)
        return res

    def run(self) -> ExecutionReport:
        loop_events: List[Dict[str, Any]] = []
        completed_steps: List[str] = []
        failed_steps: List[str] = []

        for item in self.plan.output.execution_order:
            if isinstance(item, str):
                self._current_loop_context = None
                result = self._run_step(item)
                if result.status in {"PASS", "FAIL"}:
                    completed_steps.append(item)
                if result.status == "ERROR":
                    failed_steps.append(item)
                continue

            loop = item.loop
            iterations = 0
            exit_reason = None
            loop_log_start_idx = len(self.execution_log)
            while iterations < loop.max_iterations:
                self._log(
                    f"LOOP START iteration={iterations + 1} condition=\"{loop.condition}\" steps={loop.steps}"
                )
                loop_trace = {"iteration": iterations + 1, "steps": []}
                self._current_loop_context = {
                    "in_loop": True,
                    "iteration": iterations + 1,
                    "exit_reason": None,
                }
                # Extract the condition step ID to enable early exit
                cond_step_id = None
                cond_match = re.match(r"^([A-Za-z0-9\-]+)\.", loop.condition.strip())
                if cond_match:
                    cond_step_id = cond_match.group(1)

                condition_already_met = False
                for sid in loop.steps:
                    # If the condition step already passed, skip remaining remediation steps
                    if condition_already_met:
                        self._log(f"STEP {sid} SKIPPED (loop condition already met)")
                        res = StepRunResult(
                            step_id=sid,
                            status="PASS",
                            content=f"[SKIPPED] Loop condition met at {cond_step_id}, remediation not needed.",
                            meta={"skipped_reason": "loop_condition_met"},
                        )
                        self.outputs[sid] = res
                        log_entry = self._build_log_entry(sid)
                        log_entry["actual_interaction"] = {"agent_output": res.content}
                        self.execution_log.append(log_entry)
                        loop_trace["steps"].append({"step_id": sid, "status": "SKIPPED"})
                        if sid not in completed_steps:
                            completed_steps.append(sid)
                        continue

                    result = self._run_step(sid)
                    loop_trace["steps"].append({"step_id": sid, "status": result.status})
                    if sid not in completed_steps:
                        completed_steps.append(sid)
                    if result.status == "ERROR" and sid not in failed_steps:
                        failed_steps.append(sid)

                    # Check condition right after the condition step executes
                    if sid == cond_step_id and not self._evaluate_condition(loop.condition):
                        condition_already_met = True

                loop_events.append(loop_trace)
                iterations += 1
                cond = self._evaluate_condition(loop.condition)
                self._log(f"LOOP CHECK condition_result={cond}")
                if not cond:
                    exit_reason = "condition_met"
                    break

            if exit_reason is None:
                exit_reason = "max_iterations"
            # Set exit_reason only on the last iteration's log entries (per spec)
            for entry in self.execution_log[loop_log_start_idx:]:
                if entry["loop_context"]["iteration"] == iterations:
                    entry["loop_context"]["exit_reason"] = exit_reason
            self._current_loop_context = None

        succeeded = len([s for s in failed_steps if s in self.outputs and self.outputs[s].status == "ERROR"]) == 0
        report = ExecutionReport(
            run_id=self.config.run_id,
            mode=self.config.mode,
            succeeded=succeeded,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            loop_events=loop_events,
        )
        self._write_outputs(report)
        return report

    def _write_outputs(self, report: ExecutionReport) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "events.log").write_text("\n".join(self.events), encoding="utf-8")
        (self.run_dir / "execution_report.json").write_text(
            json.dumps(report.model_dump(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        # Write structured execution log (Section 3.2 format)
        (self.run_dir / "execution_log.json").write_text(
            json.dumps(self.execution_log, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        step_payload = {k: v.model_dump() for k, v in self.outputs.items()}
        (self.run_dir / "step_outputs.json").write_text(
            json.dumps(step_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        meaningful_bundle: Dict[str, Any] = {
            "run_id": report.run_id,
            "mode": report.mode,
            "query": self.plan.input.query,
            "agents": [a.model_dump() for a in self.plan.output.agents],
            "step_results": [],
        }
        for step_id in report.completed_steps:
            if step_id not in self.outputs:
                continue
            step = self.step_defs[step_id]["step"]
            result = self.outputs[step_id]
            meaningful_bundle["step_results"].append(
                {
                    "step_id": step_id,
                    "subtask": self.step_defs[step_id]["subtask_name"],
                    "agent": self.step_defs[step_id]["agent"],
                    "objective": step.objective,
                    "instruction": step.instruction,
                    "expected_output": step.expected_output,
                    "status": result.status,
                    "tool": step.tool,
                    "actual_output": self._sanitize_text(result.content),
                }
            )

        (self.run_dir / "results_full.json").write_text(
            json.dumps(meaningful_bundle, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        lines: List[str] = []
        lines.append(f"# Live Execution Result: {report.run_id}")
        lines.append("")
        lines.append(f"- Mode: {report.mode}")
        lines.append(f"- Succeeded: {report.succeeded}")
        lines.append(f"- Completed Steps: {len(report.completed_steps)}")
        lines.append("")
        for item in meaningful_bundle["step_results"]:
            lines.append(f"## {item['step_id']} - {item['objective']}")
            lines.append(f"- Agent: {item['agent']}")
            if item["tool"]:
                lines.append(f"- Tool: {item['tool']}")
            lines.append(f"- Status: {item['status']}")
            lines.append("- Output:")
            lines.append("```text")
            lines.append(item["actual_output"])
            lines.append("```")
            lines.append("")
        (self.run_dir / "result_readable.md").write_text("\n".join(lines), encoding="utf-8")

        # Intentionally do not copy system source code into each run directory.
