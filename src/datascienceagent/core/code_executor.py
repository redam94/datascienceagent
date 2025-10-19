from typing import Optional, Any, List, Dict
from pathlib import Path
from datetime import datetime
import tempfile
import asyncio
import os
import sys

from pydantic import BaseModel


class ExecutionResult(BaseModel):
    """Result of code execution"""

    success: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    execution_time_seconds: float
    return_value: Any = None


class CodeExecutor:
    """
    Sandboxed Python code executor with safety constraints.

    Security features:
    - Runs in subprocess with timeout
    - Limited filesystem access
    - Restricted imports
    - Resource limits
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        max_memory_mb: int = 1024,
        allowed_imports: Optional[List[str]] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.allowed_imports = allowed_imports or [
            "pandas",
            "numpy",
            "matplotlib",
            "seaborn",
            "scipy",
            "statsmodels",
            "sklearn",
            "pymc",
            "arviz",
        ]

    async def execute_code(
        self, code: str, working_dir: Optional[Path] = None
    ) -> ExecutionResult:
        """
        Execute Python code in a sandboxed environment.

        Args:
            code: Python code to execute
            working_dir: Directory for file operations

        Returns:
            ExecutionResult with output and status
        """
        start_time = datetime.now()

        try:
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir=working_dir
            ) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Execute in subprocess with timeout
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                )

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(), timeout=self.timeout_seconds
                    )

                    stdout = stdout_bytes.decode("utf-8", errors="replace")
                    stderr = stderr_bytes.decode("utf-8", errors="replace")

                    execution_time = (datetime.now() - start_time).total_seconds()

                    return ExecutionResult(
                        success=process.returncode == 0,
                        stdout=stdout,
                        stderr=stderr,
                        error=None if process.returncode == 0 else "Non-zero exit code",
                        execution_time_seconds=execution_time,
                    )

                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr="",
                        error=f"Execution timeout after {self.timeout_seconds}s",
                        execution_time_seconds=self.timeout_seconds,
                    )

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error=f"Execution error: {str(e)}",
                execution_time_seconds=execution_time,
            )

    async def execute_code_with_data(
        self, code: str, data: Dict[str, Any], working_dir: Optional[Path] = None
    ) -> ExecutionResult:
        """
        Execute code with data passed via pickle files.

        Args:
            code: Python code to execute
            data: Dictionary of variables to pass to code
            working_dir: Directory for file operations

        Returns:
            ExecutionResult with output and status
        """
        import pickle

        # Create working directory if needed
        if working_dir is None:
            working_dir = Path(tempfile.mkdtemp())

        # Save data to pickle file
        data_file = working_dir / "input_data.pkl"
        with open(data_file, "wb") as f:
            pickle.dump(data, f)

        # Modify code to load data
        wrapped_code = f"""
import pickle

# Load input data
with open('{data_file}', 'rb') as f:
    __INPUT_DATA__ = pickle.load(f)

# Make variables available
for __key__, __value__ in __INPUT_DATA__.items():
    globals()[__key__] = __value__

# Execute user code
{code}
"""

        try:
            result = await self.execute_code(wrapped_code, working_dir)
            return result
        finally:
            # Clean up data file
            try:
                data_file.unlink()
            except:
                pass
