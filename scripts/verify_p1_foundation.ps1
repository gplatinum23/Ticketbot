[CmdletBinding()]
param(
    [string]$PythonExecutable = ".\\agent_env\\Scripts\\python.exe"
)

$ErrorActionPreference = "Stop"

& $PythonExecutable -m pytest -q `
    tests/test_ctrip_components.py `
    tests/test_tool_runtime.py `
    tests/test_travel_tools.py `
    tests/test_persistent_tool_cache.py `
    tests/test_p1_foundation_gates.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExecutable -m compileall -q src
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

git diff --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "P1.0-P1.4 baseline gates passed."
