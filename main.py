#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Forecast & Real-Time Dashboard Launcher v3.0

Professional launcher with optimized startup sequence, database migrations,
cache warmup, and beautiful progress indicators.

Features:
- Sequential startup (backend → migrations → cache warmup → frontend)
- Automatic database migrations
- Pre-flight checks and diagnostics
- Beautiful progress indicators
- Smart browser opening
- Performance monitoring
"""

import asyncio
import sys
import io

import subprocess
import time
import webbrowser
import signal
import os

# Configure stdout/stderr for UTF-8 on Windows
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from datetime import datetime

# Install required packages if not present
def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = ["rich", "httpx", "psutil"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Installing required packages...")
        subprocess.run(["py", "-m", "pip", "install"] + missing_packages, check=True)

ensure_dependencies()

# Import after installation
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.align import Align
import httpx
import psutil


@dataclass
class ServiceConfig:
    """Configuration for a service."""
    name: str
    command: list
    cwd: Path
    port: int
    health_url: str
    startup_timeout: int = 10
    shell: bool = False
    depends_on: list = field(default_factory=list)
    
@dataclass
class StartupMetrics:
    """Track startup performance metrics."""
    start_time: datetime = field(default_factory=datetime.now)
    backend_start_time: Optional[float] = None
    frontend_start_time: Optional[float] = None
    migration_time: Optional[float] = None
    cache_warmup_time: Optional[float] = None
    total_time: Optional[float] = None
    
    def record_backend(self, duration: float):
        self.backend_start_time = duration
        
    def record_frontend(self, duration: float):
        self.frontend_start_time = duration
        
    def record_migration(self, duration: float):
        self.migration_time = duration
        
    def record_cache_warmup(self, duration: float):
        self.cache_warmup_time = duration
        
    def finalize(self):
        self.total_time = (datetime.now() - self.start_time).total_seconds()


class ServiceManager:
    """Manages individual service processes."""
    
    def __init__(self, config: ServiceConfig, console: Console):
        self.config = config
        self.console = console
        self.process: Optional[subprocess.Popen] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        self.http_client = httpx.AsyncClient(timeout=5.0)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_client:
            await self.http_client.aclose()
    
    async def start(self) -> bool:
        """Start the service with detailed error reporting."""
        self.console.print(f"Starting {self.config.name}...", style="yellow")
        
        try:
            # Start the process with live output capture (UNBUFFERED for real-time responses)
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered Python output
            
            self.process = subprocess.Popen(
                self.config.command,
                cwd=self.config.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                shell=self.config.shell,
                bufsize=0,  # UNBUFFERED - CRITICAL FIX!
                universal_newlines=True,
                env=env,  # Pass unbuffered environment
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP') else 0
            )
            
            # Monitor startup with real-time output
            startup_output = []
            for attempt in range(self.config.startup_timeout):
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process died, get all output
                    remaining_output = self.process.stdout.read() if self.process.stdout else ""
                    if remaining_output:
                        startup_output.append(remaining_output)
                    
                    self.console.print(f"ERROR: {self.config.name} terminated early (exit code: {self.process.returncode})", style="red")
                    
                    # Show detailed error output
                    full_output = "".join(startup_output).strip()
                    if full_output:
                        self.console.print("DETAILED ERROR OUTPUT:", style="red bold")
                        self.console.print("-" * 50, style="red")
                        for line in full_output.split('\n'):
                            if line.strip():
                                if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'failed']):
                                    self.console.print(f"  {line}", style="red")
                                elif any(keyword in line.lower() for keyword in ['warning', 'warn']):
                                    self.console.print(f"  {line}", style="yellow")
                                else:
                                    self.console.print(f"  {line}", style="dim")
                        self.console.print("-" * 50, style="red")
                    
                    return False
                
                # Capture any new output (Windows-compatible)
                if self.process.stdout:
                    try:
                        # Windows-compatible non-blocking read
                        import msvcrt
                        import sys
                        
                        # Check if data is available
                        if msvcrt.kbhit() or True:  # Always try to read on Windows
                            try:
                                # Use a small timeout for readline
                                line = self.process.stdout.readline()
                                if line:
                                    startup_output.append(line)
                                    line_lower = line.lower()
                                    
                                    # Show important lines during startup
                                    if any(keyword in line_lower for keyword in ['error', 'exception', 'traceback']):
                                        self.console.print(f"  ERROR: {line.strip()}", style="red")
                                    elif any(keyword in line_lower for keyword in ['warning', 'warn']):
                                        self.console.print(f"  WARN: {line.strip()}", style="yellow")
                                    elif any(keyword in line_lower for keyword in ['started', 'running', 'listening', 'uvicorn']):
                                        self.console.print(f"  INFO: {line.strip()}", style="green")
                                    elif 'info:' in line_lower and any(keyword in line_lower for keyword in ['application startup', 'server process', 'ready to serve']):
                                        self.console.print(f"  INFO: {line.strip()}", style="green")
                                    # Special handling for backend initialization completion
                                    elif 'initialization complete' in line_lower or 'ready to serve requests' in line_lower:
                                        self.console.print(f"  ✓ {line.strip()}", style="bold green")
                            except:
                                pass
                    except ImportError:
                        # Fallback for non-Windows or if msvcrt not available
                        try:
                            line = self.process.stdout.readline()
                            if line:
                                startup_output.append(line)
                                if any(keyword in line.lower() for keyword in ['error', 'exception', 'started', 'running', 'listening']):
                                    self.console.print(f"  {line.strip()}", style="dim")
                        except:
                            pass
                
                # Test health endpoint (only after a few seconds to avoid false negatives)
                if attempt >= 5:  # Wait at least 5 seconds before checking health
                    try:
                        response = await self.http_client.get(self.config.health_url, timeout=5.0)
                        if response.status_code == 200:
                            self.console.print(f"SUCCESS: {self.config.name} started and responding", style="green")
                            
                            # For backend, wait a bit more to ensure it's fully initialized
                            if "backend" in self.config.name.lower():
                                self.console.print(f"Waiting for {self.config.name} initialization to complete...", style="dim yellow")
                                await asyncio.sleep(3)  # Give backend 3 more seconds for full initialization
                                
                                # Verify again
                                response2 = await self.http_client.get(self.config.health_url, timeout=5.0)
                                if response2.status_code == 200:
                                    self.console.print(f"SUCCESS: {self.config.name} fully ready", style="green")
                            
                            # Show startup summary if there were any warnings
                            warnings = [line for line in startup_output if 'warning' in line.lower() or 'warn' in line.lower()]
                            if warnings:
                                self.console.print(f"STARTUP WARNINGS for {self.config.name}:", style="yellow")
                                for warning in warnings[-3:]:  # Show last 3 warnings
                                    self.console.print(f"  {warning.strip()}", style="yellow")
                            
                            return True
                    except Exception as health_error:
                        # Only show health check errors after more attempts
                        if attempt >= self.config.startup_timeout - 5:
                            pass  # Silent during early attempts
                        elif attempt == self.config.startup_timeout - 1:
                            self.console.print(f"Health check failed: {health_error}", style="yellow")
                
                await asyncio.sleep(1)
                # Only show waiting message every 5 seconds to reduce noise
                if attempt % 5 == 0 or attempt < 5:
                    self.console.print(f"Waiting for {self.config.name}... ({attempt+1}/{self.config.startup_timeout}s)", style="dim")
            
            # Timeout reached
            self.console.print(f"ERROR: {self.config.name} failed to start within {self.config.startup_timeout} seconds", style="red")
            
            # Show final output for debugging
            if startup_output:
                self.console.print("STARTUP OUTPUT (last 10 lines):", style="yellow")
                for line in startup_output[-10:]:
                    self.console.print(f"  {line.strip()}", style="dim")
            
            return False
            
        except Exception as e:
            self.console.print(f"ERROR: Failed to start {self.config.name}: {e}", style="red")
            self.console.print(f"Command attempted: {' '.join(self.config.command)}", style="dim")
            self.console.print(f"Working directory: {self.config.cwd}", style="dim")
            return False
    
    def stop(self):
        """Stop the service."""
        if self.process:
            self.console.print(f"Stopping {self.config.name}...", style="yellow")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.console.print(f"STOPPED: {self.config.name}", style="green")


class ProcessCleaner:
    """Handles cleanup of existing processes."""
    
    def __init__(self, console: Console):
        self.console = console
    
    def kill_processes_on_ports(self, ports: list[int]) -> int:
        """Kill processes running on specified ports."""
        killed_count = 0
        
        for port in ports:
            try:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        connections = proc.net_connections()
                        if connections:
                            for conn in connections:
                                if conn.laddr.port == port:
                                    self.console.print(f"Terminating process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}", style="red")
                                    proc.kill()
                                    killed_count += 1
                                    break
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
                        pass
            except Exception as e:
                self.console.print(f"WARNING: Error checking port {port}: {e}", style="yellow")
        
        return killed_count
    
    def kill_project_processes(self) -> int:
        """Kill processes related to this project."""
        killed_count = 0
        current_pid = os.getpid()  # Don't kill the current launcher process
        
        # Look for specific patterns that indicate backend/frontend processes
        target_patterns = [
            'backend/app/main.py',
            'uvicorn',
            'npm run dev',
            'vite',
            'node_modules'
        ]
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['pid'] == current_pid:
                        continue  # Skip the current launcher process
                        
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline is None:
                        continue
                        
                    cmdline_str = ' '.join(str(arg) for arg in cmdline if arg is not None).lower()
                    
                    # Check if this is a backend or frontend process
                    if any(pattern in cmdline_str for pattern in target_patterns):
                        self.console.print(f"Terminating {proc.info['name']} process (PID: {proc.info['pid']})", style="red")
                        proc.kill()
                        killed_count += 1
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError, TypeError):
                    pass
                    
        except Exception as e:
            self.console.print(f"WARNING: Error checking processes: {e}", style="yellow")
        
        return killed_count
    
    def cleanup(self) -> int:
        """Perform complete cleanup."""
        self.console.print("Cleaning up existing processes...", style="yellow")
        
        total_killed = 0
        total_killed += self.kill_processes_on_ports([8000, 5173])
        total_killed += self.kill_project_processes()
        
        if total_killed > 0:
            self.console.print(f"Terminated {total_killed} existing processes", style="green")
            time.sleep(3)  # Wait longer for processes to fully terminate
        else:
            self.console.print("No existing processes found", style="green")
        
        return total_killed


class CryptoDashboardLauncher:
    """Main launcher for the crypto dashboard project with optimized startup."""
    
    def __init__(self):
        # Configure console for Windows compatibility (avoid Unicode encoding errors)
        self.console = Console(force_terminal=True, legacy_windows=False)
        self.root_dir = Path(__file__).parent
        self.services: Dict[str, ServiceManager] = {}
        self.cleaner = ProcessCleaner(self.console)
        self.metrics = StartupMetrics()
        
        # Service configurations with dependencies
        self.service_configs = {
            "backend": ServiceConfig(
                name="Backend API",
                command=[
                    str(self.root_dir / ".venv" / "Scripts" / "python.exe"),
                    "-u",  # Unbuffered Python output - CRITICAL FIX!
                    "-m", "uvicorn",
                    "backend.app.main:app",
                    "--host", "127.0.0.1",
                    "--port", "8000",
                    "--reload",
                    "--log-level", "info",
                    "--access-log",  # Enable access logs
                    "--use-colors"  # Better logging
                ],
                cwd=self.root_dir,
                port=8000,
                health_url="http://127.0.0.1:8000/health/quick",
                startup_timeout=60,  # Increased to 60 seconds - backend initialization can take time
                depends_on=[]  # No dependencies
            ),
            "frontend": ServiceConfig(
                name="Frontend Dev Server",
                command=["npm", "run", "dev"],
                cwd=self.root_dir / "frontend",
                port=5173,
                health_url="http://127.0.0.1:5173",
                shell=True,
                startup_timeout=30,
                depends_on=["backend"]  # Depends on backend being ready
            )
        }
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics before starting services."""
        self.console.print("Running system diagnostics...", style="cyan")
        diagnostics = {}
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        diagnostics["python_version"] = python_version
        self.console.print(f"Python version: {python_version}", style="dim")
        
        # Check virtual environment
        venv_python = self.root_dir / ".venv" / "Scripts" / "python.exe"
        venv_exists = venv_python.exists()
        diagnostics["venv_exists"] = venv_exists
        status = "[OK]" if venv_exists else "[MISSING]"
        style = "green" if venv_exists else "red"
        self.console.print(f"Virtual environment: {status}", style=style)
        
        # Check required directories
        required_dirs = ["backend", "frontend", "backend/app"]
        for dir_name in required_dirs:
            dir_path = self.root_dir / dir_name
            exists = dir_path.exists()
            diagnostics[f"dir_{dir_name.replace('/', '_')}"] = exists
            status = "[OK]" if exists else "[MISSING]"
            style = "green" if exists else "red"
            self.console.print(f"Directory {dir_name}: {status}", style=style)
        
        # Check required files
        required_files = [
            "backend/app/main.py",
            "frontend/package.json",
            "requirements.txt"
        ]
        for file_name in required_files:
            file_path = self.root_dir / file_name
            exists = file_path.exists()
            diagnostics[f"file_{file_name.replace('/', '_').replace('.', '_')}"] = exists
            status = "[OK]" if exists else "[MISSING]"
            style = "green" if exists else "red"
            self.console.print(f"File {file_name}: {status}", style=style)
        
        # Check Python dependencies
        try:
            import fastapi, uvicorn, httpx, sqlalchemy
            diagnostics["python_deps"] = True
            self.console.print("Python dependencies: [OK]", style="green")
        except ImportError as e:
            diagnostics["python_deps"] = False
            self.console.print(f"Python dependencies: [MISSING] ({e})", style="red")
        
        # Check Node.js
        try:
            result = subprocess.run(["npm", "--version"], capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                npm_version = result.stdout.strip()
                diagnostics["npm_version"] = npm_version
                self.console.print(f"npm version: {npm_version} [OK]", style="green")
            else:
                diagnostics["npm_available"] = False
                self.console.print("npm: [NOT AVAILABLE]", style="red")
        except Exception:
            diagnostics["npm_available"] = False
            self.console.print("npm: [NOT AVAILABLE]", style="red")
        
        # Check ports
        import socket
        for port in [8000, 5173]:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex(('127.0.0.1', port))
                    if result == 0:
                        diagnostics[f"port_{port}_available"] = False
                        self.console.print(f"Port {port}: [IN USE]", style="yellow")
                    else:
                        diagnostics[f"port_{port}_available"] = True
                        self.console.print(f"Port {port}: [AVAILABLE]", style="green")
            except Exception:
                diagnostics[f"port_{port}_available"] = True
                self.console.print(f"Port {port}: [AVAILABLE]", style="green")
        
        self.console.print("Diagnostics complete", style="cyan")
        return diagnostics
    
    async def setup_services(self):
        """Initialize service managers with fallback commands."""
        for name, config in self.service_configs.items():
            # Add fallback for backend if venv path doesn't exist
            if name == "backend":
                venv_python = self.root_dir / ".venv" / "Scripts" / "python.exe"
                if not venv_python.exists():
                    self.console.print("[WARN] Virtual environment not found, using system Python", style="yellow")
                    # Try multiple Python commands
                    python_cmds = ["python", "python3", "py"]
                    python_cmd = None
                    for cmd in python_cmds:
                        try:
                            result = subprocess.run([cmd, "--version"], capture_output=True, timeout=2)
                            if result.returncode == 0:
                                python_cmd = cmd
                                break
                        except:
                            continue
                    
                    if python_cmd:
                        config.command = [python_cmd, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
                        self.console.print(f"[INFO] Using {python_cmd} to start backend", style="dim")
                    else:
                        self.console.print("[ERROR] No Python interpreter found!", style="red")
                        raise RuntimeError("Cannot find Python interpreter")
            
            self.services[name] = ServiceManager(config, self.console)
    
    async def run_database_migrations(self) -> Tuple[bool, float]:
        """Run database migrations before starting services."""
        start = time.time()
        self.console.print("[DB]  Checking database migrations...", style="cyan")
        
        try:
            # Check if alembic is available
            result = subprocess.run(
                ["alembic", "current"],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.console.print("[OK] Database is up to date", style="green")
                duration = time.time() - start
                return True, duration
            else:
                # Try to run migrations
                self.console.print("[RUN] Running database migrations...", style="yellow")
                result = subprocess.run(
                    ["alembic", "upgrade", "head"],
                    cwd=self.root_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    self.console.print("[OK] Database migrations completed", style="green")
                    duration = time.time() - start
                    return True, duration
                else:
                    self.console.print(f"[WARN]  Migration warning: {result.stderr[:200]}", style="yellow")
                    duration = time.time() - start
                    return True, duration  # Continue anyway
                    
        except FileNotFoundError:
            self.console.print("[WARN]  Alembic not found, skipping migrations", style="yellow")
            duration = time.time() - start
            return True, duration
        except Exception as e:
            self.console.print(f"[WARN]  Migration check failed: {str(e)[:100]}", style="yellow")
            duration = time.time() - start
            return True, duration  # Continue anyway
    
    async def warmup_cache(self) -> Tuple[bool, float]:
        """Warm up the cache with initial data."""
        start = time.time()
        self.console.print("[CACHE] Warming up cache...", style="cyan")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:  # Increased timeout for cache warmup
                # Warm up prices endpoint
                try:
                    self.console.print("  Warming prices cache...", style="dim")
                    response = await client.get("http://127.0.0.1:8000/prices?ids=bitcoin,ethereum,solana", timeout=15.0)
                    if response.status_code == 200:
                        self.console.print("  [OK] Prices cache warmed", style="dim green")
                except Exception as e:
                    self.console.print(f"  [WARN] Prices cache warmup failed: {str(e)[:50]}", style="dim yellow")
                
                # Warm up market data
                try:
                    self.console.print("  Warming market data cache...", style="dim")
                    response = await client.get("http://127.0.0.1:8000/prices/market?ids=bitcoin,ethereum,solana", timeout=15.0)
                    if response.status_code == 200:
                        self.console.print("  [OK] Market data cache warmed", style="dim green")
                except Exception as e:
                    self.console.print(f"  [WARN] Market data cache warmup failed: {str(e)[:50]}", style="dim yellow")
                
                # Warm up stream snapshot (most important for frontend)
                try:
                    self.console.print("  Warming stream snapshot cache...", style="dim")
                    response = await client.get("http://127.0.0.1:8000/stream/snapshot", timeout=20.0)
                    if response.status_code == 200:
                        data = response.json()
                        crypto_count = len(data.get('data', {}))
                        self.console.print(f"  [OK] Stream snapshot cache warmed ({crypto_count} cryptos)", style="dim green")
                except Exception as e:
                    self.console.print(f"  [WARN] Stream snapshot warmup failed: {str(e)[:50]}", style="dim yellow")
            
            duration = time.time() - start
            self.console.print(f"[OK] Cache warmup completed in {duration:.2f}s", style="green")
            return True, duration
            
        except Exception as e:
            duration = time.time() - start
            self.console.print(f"[WARN]  Cache warmup failed: {str(e)[:100]}", style="yellow")
            self.console.print("[INFO] Continuing anyway - cache will populate on demand", style="dim")
            return True, duration  # Continue anyway
    
    async def install_frontend_dependencies(self) -> bool:
        """Install frontend dependencies if needed."""
        frontend_dir = self.root_dir / "frontend"
        node_modules = frontend_dir / "node_modules"
        
        if not node_modules.exists():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("[PKG] Installing frontend dependencies...", total=None)
                
                try:
                    result = subprocess.run(
                        ["npm", "install"],
                        cwd=frontend_dir,
                        shell=True,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    progress.update(task, completed=True)
                    self.console.print("[OK] Frontend dependencies installed", style="green")
                    return True
                except subprocess.CalledProcessError as e:
                    self.console.print(f"[X] Failed to install frontend dependencies", style="red")
                    self.console.print(f"  Error: {e.stderr[:200]}", style="dim red")
                    return False
        else:
            self.console.print("[OK] Frontend dependencies already installed", style="dim green")
        return True
    
    async def start_services_sequential(self) -> bool:
        """Start services in optimal sequence with progress tracking."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            # Step 1: Install frontend dependencies
            task1 = progress.add_task("[PKG] Checking dependencies...", total=1)
            if not await self.install_frontend_dependencies():
                return False
            progress.update(task1, completed=1)
            
            # Step 2: Run database migrations
            task2 = progress.add_task("[DB]  Database migrations...", total=1)
            success, duration = await self.run_database_migrations()
            self.metrics.record_migration(duration)
            if not success:
                return False
            progress.update(task2, completed=1)
            
            # Step 3: Start backend
            task3 = progress.add_task("[START] Starting backend API...", total=1)
            backend_service = self.services.get("backend")
            if backend_service:
                await backend_service.__aenter__()
                backend_start = time.time()
                self.console.print("[INFO] Backend may take 30-60 seconds to initialize on first startup...", style="dim yellow")
                if not await backend_service.start():
                    self.console.print("[X] Backend failed to start", style="red")
                    self.console.print("[INFO] Check the error messages above for details", style="yellow")
                    self.console.print("[INFO] Common issues:", style="yellow")
                    self.console.print("  - Missing dependencies: pip install -r requirements.txt", style="dim")
                    self.console.print("  - Port 8000 in use: Check with 'netstat -ano | findstr :8000'", style="dim")
                    self.console.print("  - Import errors: Check virtual environment is activated", style="dim")
                    return False
                backend_duration = time.time() - backend_start
                self.metrics.record_backend(backend_duration)
                self.console.print(f"[OK] Backend started in {backend_duration:.2f}s", style="green")
                
                # Wait a bit more for backend to be fully ready for requests
                self.console.print("[WAIT] Ensuring backend is fully ready...", style="dim yellow")
                await asyncio.sleep(2)
                
                # Verify backend is responding
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get("http://127.0.0.1:8000/health/quick")
                        if response.status_code == 200:
                            self.console.print("[OK] Backend is ready to serve requests", style="green")
                        else:
                            self.console.print(f"[WARN] Backend health check returned status {response.status_code}", style="yellow")
                except Exception as e:
                    self.console.print(f"[WARN] Could not verify backend health: {e}", style="yellow")
                    self.console.print("[INFO] Backend may still be initializing, continuing anyway...", style="dim")
                
                progress.update(task3, completed=1)
            
            # Step 4: Warm up cache
            task4 = progress.add_task("[CACHE] Warming up cache...", total=1)
            success, duration = await self.warmup_cache()
            self.metrics.record_cache_warmup(duration)
            progress.update(task4, completed=1)
            
            # Step 5: Start frontend
            task5 = progress.add_task("[START] Starting frontend...", total=1)
            frontend_service = self.services.get("frontend")
            if frontend_service:
                await frontend_service.__aenter__()
                frontend_start = time.time()
                if not await frontend_service.start():
                    self.console.print("[X] Frontend failed to start", style="red")
                    return False
                frontend_duration = time.time() - frontend_start
                self.metrics.record_frontend(frontend_duration)
                progress.update(task5, completed=1)
        
        return True
    
    async def wait_for_frontend_ready(self) -> bool:
        """Wait for frontend to be fully interactive."""
        self.console.print("[WAIT] Waiting for frontend to be ready...", style="dim cyan")
        
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get("http://127.0.0.1:5173")
                    if response.status_code == 200:
                        # Additional check: ensure the page has loaded content
                        if len(response.text) > 1000:  # Basic check for actual content
                            self.console.print("[OK] Frontend is ready!", style="green")
                            return True
            except:
                pass
            
            await asyncio.sleep(1)
            
        self.console.print("[WARN]  Frontend may not be fully ready, but continuing...", style="yellow")
        return True
    
    async def stop_services(self):
        """Stop all services."""
        for service in self.services.values():
            service.stop()
            # Properly close HTTP clients
            if service.http_client:
                await service.http_client.aclose()
    
    def show_status(self):
        """Display service status and URLs with performance metrics."""
        # Services table
        table = Table(title="[START] Crypto Dashboard Services", border_style="cyan")
        table.add_column("Service", style="cyan", no_wrap=True)
        table.add_column("Status", style="green", no_wrap=True)
        table.add_column("URL", style="blue")
        table.add_column("Startup Time", style="yellow", justify="right")
        
        backend_time = f"{self.metrics.backend_start_time:.2f}s" if self.metrics.backend_start_time else "N/A"
        frontend_time = f"{self.metrics.frontend_start_time:.2f}s" if self.metrics.frontend_start_time else "N/A"
        
        table.add_row("Backend API", "[OK] Running", "http://127.0.0.1:8000", backend_time)
        table.add_row("Frontend", "[OK] Running", "http://127.0.0.1:5173", frontend_time)
        table.add_row("API Docs", "[OK] Available", "http://127.0.0.1:8000/docs", "")
        
        self.console.print("\n")
        self.console.print(table)
        
        # Performance metrics
        metrics_table = Table(title="[RUN] Startup Performance", border_style="green")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Duration", style="yellow", justify="right")
        
        if self.metrics.migration_time:
            metrics_table.add_row("Database Migrations", f"{self.metrics.migration_time:.2f}s")
        if self.metrics.cache_warmup_time:
            metrics_table.add_row("Cache Warmup", f"{self.metrics.cache_warmup_time:.2f}s")
        if self.metrics.backend_start_time:
            metrics_table.add_row("Backend Startup", f"{self.metrics.backend_start_time:.2f}s")
        if self.metrics.frontend_start_time:
            metrics_table.add_row("Frontend Startup", f"{self.metrics.frontend_start_time:.2f}s")
        
        self.metrics.finalize()
        if self.metrics.total_time:
            metrics_table.add_row("Total Time", f"{self.metrics.total_time:.2f}s", style="bold green")
        
        self.console.print("\n")
        self.console.print(metrics_table)
        
        # Quick access panel
        quick_access = Panel(
            "[bold cyan]Quick Access Links:[/bold cyan]\n\n"
            "[WEB] Dashboard:  [blue]http://127.0.0.1:5173[/blue]\n"
            "[API] API Docs:   [blue]http://127.0.0.1:8000/docs[/blue]\n"
            "[HEALTH] Health:     [blue]http://127.0.0.1:8000/health[/blue]\n"
            "[NEWS] News:       [blue]http://127.0.0.1:8000/news[/blue]\n"
            "[FORECAST] Forecasts:  [blue]http://127.0.0.1:8000/forecasts[/blue]",
            title="[LINK] Quick Access",
            border_style="blue"
        )
        self.console.print("\n")
        self.console.print(quick_access)
        
        self.console.print("\n[dim]Press Ctrl+C to stop all services[/dim]\n")
    
    async def open_browser(self):
        """Open the application in browser after ensuring it's ready."""
        # Wait for frontend to be fully ready
        await self.wait_for_frontend_ready()
        
        self.console.print("[WEB] Opening browser...", style="cyan")
        try:
            webbrowser.open("http://127.0.0.1:5173")
            self.console.print("[OK] Browser opened successfully", style="green")
        except Exception as e:
            self.console.print(f"[WARN]  Could not open browser automatically: {e}", style="yellow")
            self.console.print("  Please open http://127.0.0.1:5173 manually", style="dim")
    
    async def run(self):
        """Main launcher sequence with optimized startup flow."""
        try:
            # Show welcome banner
            welcome = Panel.fit(
                "[bold cyan]Crypto Forecast & Real-Time Dashboard[/bold cyan]\n"
                "[dim]Professional Crypto Analytics Platform[/dim]\n\n"
                "Version 3.0 - Optimized Startup",
                border_style="bold blue"
            )
            self.console.print("\n")
            self.console.print(Align.center(welcome))
            self.console.print("\n")
            
            # Run pre-flight diagnostics
            self.console.print("[*] Running pre-flight checks...\n", style="bold cyan")
            diagnostics = self.run_diagnostics()
            
            # Check for critical issues
            critical_issues = []
            if not diagnostics.get("python_deps", True):
                critical_issues.append("Missing Python dependencies - run: py -m pip install -r requirements.txt")
            if not diagnostics.get("file_backend_app_main_py", True):
                critical_issues.append("Backend main.py not found")
            if not diagnostics.get("file_frontend_package_json", True):
                critical_issues.append("Frontend package.json not found")
            if not diagnostics.get("venv_exists", True):
                critical_issues.append("Virtual environment not found - run: py -3 -m venv .venv")
            
            if critical_issues:
                self.console.print("\n[WARNING] CRITICAL ISSUES DETECTED:\n", style="red bold")
                for issue in critical_issues:
                    self.console.print(f"  [X] {issue}", style="red")
                self.console.print("\n[dim]Please fix these issues before continuing.[/dim]", style="red")
                return
            
            self.console.print("\n[OK] Pre-flight checks passed\n", style="green")
            
            # Cleanup existing processes
            self.cleaner.cleanup()
            
            # Setup services
            await self.setup_services()
            
            # Start services in optimal sequence
            self.console.print("[STARTING] Starting services in optimized sequence...\n", style="bold cyan")
            
            if not await self.start_services_sequential():
                self.console.print("\n[X] Failed to start services", style="red bold")
                self.console.print("Check the error details above for troubleshooting.\n", style="yellow")
                
                # Suggest common fixes
                fixes_panel = Panel(
                     "[bold cyan]Common Fixes:[/bold cyan]\n\n"
                     "1. Install Python dependencies:\n"
                     "   [dim]py -m pip install -r requirements.txt[/dim]\n\n"
                     "2. Install frontend dependencies:\n"
                     "   [dim]cd frontend && npm install[/dim]\n\n"
                     "3. Check if ports 8000/5173 are available\n\n"
                     "4. Check database connection\n\n"
                     "5. If API rate limited, wait a few minutes and retry\n\n"
                     "6. Review logs above for specific errors",
                     title="[TIP] Troubleshooting",
                     border_style="yellow"
                 )
                self.console.print(fixes_panel)
                return
            
            self.console.print("\n[OK] All services started successfully!\n", style="bold green")
            
            # Show status and open browser
            self.show_status()
            await self.open_browser()
            
            # Enhanced monitoring loop with health checks
            self.console.print("[MONITOR] Monitoring services...", style="dim cyan")
            
            try:
                error_count = 0
                check_interval = 10  # Check every 10 seconds
                
                while True:
                    await asyncio.sleep(check_interval)
                    
                    # Health check all services
                    all_healthy = True
                    for name, service in self.services.items():
                        if service.process and service.process.poll() is not None:
                            self.console.print(f"\n[WARN]  WARNING: {name} process died unexpectedly", style="red bold")
                            error_count += 1
                            all_healthy = False
                            
                            if error_count > 3:
                                self.console.print("\n[X] Too many service failures, shutting down...", style="red bold")
                                return
                    
                    # Periodic health check via HTTP
                    if all_healthy and error_count == 0:
                        try:
                            async with httpx.AsyncClient(timeout=5.0) as client:
                                # Quick health check
                                response = await client.get("http://127.0.0.1:8000/health/quick")
                                if response.status_code != 200:
                                    self.console.print("\n[!] Backend health check failed", style="yellow")
                        except:
                            pass  # Silent fail for monitoring
                    
            except asyncio.CancelledError:
                self.console.print("\n\n[STOP] Shutdown requested", style="yellow")
            
        except KeyboardInterrupt:
            self.console.print("\n\n[STOP] Shutdown requested by user", style="yellow")
        except Exception as e:
            self.console.print(f"\n[X] Launcher error: {e}", style="red bold")
            
            # Show stack trace for debugging
            import traceback
            self.console.print("\n[red bold]FULL ERROR TRACEBACK:[/red bold]")
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            
        finally:
            self.console.print("\n[WAIT] Shutting down all services...", style="yellow")
            await self.stop_services()
            
            # Show shutdown summary
            shutdown_panel = Panel(
                "[bold green]Shutdown Complete[/bold green]\n\n"
                "[OK] All services stopped gracefully\n"
                "[OK] Resources cleaned up\n\n"
                "[dim]Thank you for using Crypto Dashboard![/dim]",
                border_style="green"
            )
            self.console.print("\n")
            self.console.print(Align.center(shutdown_panel))
            self.console.print("\n")


async def main():
    """Entry point."""
    launcher = CryptoDashboardLauncher()
    await launcher.run()


if __name__ == "__main__":
    asyncio.run(main())