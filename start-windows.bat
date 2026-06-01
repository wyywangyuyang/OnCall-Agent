@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ====================================
echo 启动 OnCall-Agent 服务
echo ====================================
echo.

REM 设置 Python 命令
set PYTHON_CMD=.venv\Scripts\python.exe

REM 检查虚拟环境是否存在
if not exist .venv\Scripts\python.exe (
    echo [错误] 虚拟环境不存在，请先运行完整初始化脚本
    pause
    exit /b 1
)

REM 启动 Docker Compose
echo [1/4] 启动 Milvus 向量数据库...
docker ps --format "{{.Names}}" | findstr "milvus-standalone" >nul 2>&1
if not errorlevel 1 (
    echo [信息] Milvus 容器已在运行
) else (
    docker compose -f vector-database.yml up -d
    if errorlevel 1 (
        echo [错误] Docker 启动失败，请确保 Docker Desktop 已启动
        pause
        exit /b 1
    )
    echo [信息] 等待 Milvus 启动（10秒）...
    timeout /t 10 /nobreak >nul
)
echo [成功] Milvus 数据库就绪
echo.

REM 启动 CLS MCP 服务
echo [2/4] 启动 CLS MCP 服务...
start "CLS MCP Server" /min %PYTHON_CMD% mcp_servers/cls_server.py
timeout /t 2 /nobreak >nul
echo [成功] CLS MCP 服务已启动
echo.

REM 启动 Monitor MCP 服务
echo [3/4] 启动 Monitor MCP 服务...
start "Monitor MCP Server" /min %PYTHON_CMD% mcp_servers/monitor_server.py
timeout /t 2 /nobreak >nul
echo [成功] Monitor MCP 服务已启动
echo.

REM 启动 FastAPI 服务
echo [4/4] 启动 FastAPI 服务...
start "OnCall-Agent API" cmd /k "%PYTHON_CMD% -m uvicorn app.main:app --host 0.0.0.0 --port 9900"
echo [信息] 等待服务启动（15秒）...
timeout /t 15 /nobreak >nul
echo.

REM 检查服务状态并上传文档
echo.
echo [信息] 检查服务状态...
curl -s http://localhost:9900/health >nul 2>&1
if errorlevel 1 (
    echo [警告] 服务可能还未完全启动，请稍等片刻
) else (
    echo [成功] FastAPI 服务运行正常
    echo.

    REM 调用 API 上传 aiops-docs 文档到向量数据库
    echo [信息] 上传文档到向量数据库...
    for %%f in (aiops-docs\*.md) do (
        echo   上传: %%~nxf
        curl -s -X POST http://localhost:9900/api/upload -F "file=@%%f" >nul 2>&1
    )
    echo [成功] 文档上传完成
)

echo.
echo ====================================
echo 服务启动完成！
echo ====================================
echo Web 界面: http://localhost:9900
echo API 文档: http://localhost:9900/docs
echo Milvus 管理工具 (Attu): http://localhost:8000
echo.
echo 查看日志:
echo   - FastAPI: logs\app_*.log（Loguru 日志，按天轮转）
echo   - CLS MCP: mcp_cls.log（当前目录）
echo   - Monitor: mcp_monitor.log（当前目录）
echo 停止服务: stop-windows.bat
echo ====================================
pause
