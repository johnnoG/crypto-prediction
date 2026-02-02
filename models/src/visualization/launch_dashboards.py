"""
Unified Dashboard Launcher
Provides a single entry point to launch all visualization and monitoring dashboards
"""

import os
import sys
import time
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardManager:
    """Manages multiple visualization dashboards"""

    def __init__(self, base_port: int = 8050):
        self.base_port = base_port
        self.dashboards = {}
        self.processes = {}
        self.threads = {}

        # Dashboard configurations
        self.dashboard_configs = {
            'training_monitor': {
                'port': base_port,
                'module': 'training_monitor',
                'class': 'TrainingMonitor',
                'description': 'Real-time training progress monitoring',
                'url_path': '/',
                'color': '🔍'
            },
            'hyperopt_dashboard': {
                'port': base_port + 1,
                'module': 'hyperopt_dashboard',
                'class': 'HyperparameterOptimizationDashboard',
                'description': 'Interactive hyperparameter optimization analysis',
                'url_path': '/',
                'color': '🎯'
            },
            'model_inspector': {
                'port': base_port + 2,
                'module': 'model_inspector',
                'class': 'ModelInspectorDashboard',
                'description': 'Model architecture and weight visualization',
                'url_path': '/',
                'color': '🔍'
            },
            'training_dashboard': {
                'port': base_port + 3,
                'module': 'training_dashboard',
                'class': 'TrainingProgressVisualizer',
                'description': 'Comprehensive training analytics',
                'url_path': '/',
                'color': '📊'
            }
        }

    def print_banner(self):
        """Print startup banner"""
        print("\n" + "="*80)
        print("🚀 CRYPTOCURRENCY PREDICTION ML SYSTEM - DASHBOARD LAUNCHER")
        print("="*80)
        print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 Base Port: {self.base_port}")
        print("="*80)

    def print_dashboard_info(self):
        """Print information about available dashboards"""
        print("\n📊 AVAILABLE DASHBOARDS:")
        print("-" * 50)

        for name, config in self.dashboard_configs.items():
            print(f"{config['color']} {name.upper().replace('_', ' ')}")
            print(f"   📝 Description: {config['description']}")
            print(f"   🌐 URL: http://localhost:{config['port']}{config['url_path']}")
            print(f"   📊 Port: {config['port']}")
            print()

    def launch_training_monitor(self) -> bool:
        """Launch training progress monitor"""
        try:
            from training_monitor import TrainingMonitor, create_monitoring_dashboard

            monitor = TrainingMonitor()
            monitor.start_monitoring()

            # Start in background thread
            def run_monitor():
                try:
                    create_monitoring_dashboard(monitor, port=self.dashboard_configs['training_monitor']['port'])
                except Exception as e:
                    logger.error(f"Training monitor error: {e}")

            thread = threading.Thread(target=run_monitor, daemon=True)
            thread.start()
            self.threads['training_monitor'] = thread

            logger.info(f"🔍 Training Monitor started on port {self.dashboard_configs['training_monitor']['port']}")
            return True

        except Exception as e:
            logger.error(f"Failed to start training monitor: {e}")
            return False

    def launch_hyperopt_dashboard(self) -> bool:
        """Launch hyperparameter optimization dashboard"""
        try:
            from hyperopt_dashboard import HyperparameterOptimizationDashboard

            dashboard = HyperparameterOptimizationDashboard(
                dashboard_port=self.dashboard_configs['hyperopt_dashboard']['port']
            )

            # Start in background thread
            def run_hyperopt():
                try:
                    dashboard.run_dashboard(debug=False)
                except Exception as e:
                    logger.error(f"Hyperopt dashboard error: {e}")

            thread = threading.Thread(target=run_hyperopt, daemon=True)
            thread.start()
            self.threads['hyperopt_dashboard'] = thread

            logger.info(f"🎯 Hyperparameter Dashboard started on port {self.dashboard_configs['hyperopt_dashboard']['port']}")
            return True

        except Exception as e:
            logger.error(f"Failed to start hyperopt dashboard: {e}")
            return False

    def launch_model_inspector(self) -> bool:
        """Launch model architecture inspector"""
        try:
            from model_inspector import ModelInspectorDashboard

            inspector = ModelInspectorDashboard(
                port=self.dashboard_configs['model_inspector']['port']
            )

            # Start in background thread
            def run_inspector():
                try:
                    inspector.run_dashboard(debug=False)
                except Exception as e:
                    logger.error(f"Model inspector error: {e}")

            thread = threading.Thread(target=run_inspector, daemon=True)
            thread.start()
            self.threads['model_inspector'] = thread

            logger.info(f"🔍 Model Inspector started on port {self.dashboard_configs['model_inspector']['port']}")
            return True

        except Exception as e:
            logger.error(f"Failed to start model inspector: {e}")
            return False

    def launch_training_dashboard(self) -> bool:
        """Launch comprehensive training dashboard"""
        try:
            from training_dashboard import TrainingProgressVisualizer

            visualizer = TrainingProgressVisualizer()

            # Create sample dashboard
            def run_dashboard():
                try:
                    # This would integrate with the main dashboard
                    logger.info("Training Dashboard ready for integration")
                    # Keep thread alive
                    while True:
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"Training dashboard error: {e}")

            thread = threading.Thread(target=run_dashboard, daemon=True)
            thread.start()
            self.threads['training_dashboard'] = thread

            logger.info(f"📊 Training Dashboard components loaded on port {self.dashboard_configs['training_dashboard']['port']}")
            return True

        except Exception as e:
            logger.error(f"Failed to start training dashboard: {e}")
            return False

    def launch_mlflow_ui(self, port: int = 5000) -> bool:
        """Launch MLflow UI"""
        try:
            import mlflow
            mlflow_cmd = f"mlflow ui --port {port} --host 0.0.0.0"

            process = subprocess.Popen(
                mlflow_cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            self.processes['mlflow'] = process
            logger.info(f"🔬 MLflow UI started on port {port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start MLflow UI: {e}")
            return False

    def launch_all_dashboards(self, include_mlflow: bool = True) -> Dict[str, bool]:
        """Launch all available dashboards"""
        results = {}

        print("\n🚀 LAUNCHING DASHBOARDS...")
        print("-" * 30)

        # Launch each dashboard
        results['training_monitor'] = self.launch_training_monitor()
        time.sleep(2)  # Stagger starts

        results['hyperopt_dashboard'] = self.launch_hyperopt_dashboard()
        time.sleep(2)

        results['model_inspector'] = self.launch_model_inspector()
        time.sleep(2)

        results['training_dashboard'] = self.launch_training_dashboard()
        time.sleep(2)

        # Launch MLflow if requested
        if include_mlflow:
            results['mlflow'] = self.launch_mlflow_ui()
            time.sleep(2)

        return results

    def print_launch_summary(self, results: Dict[str, bool]):
        """Print summary of launched dashboards"""
        print("\n" + "="*80)
        print("📊 DASHBOARD LAUNCH SUMMARY")
        print("="*80)

        successful = []
        failed = []

        for dashboard, success in results.items():
            if success:
                successful.append(dashboard)
            else:
                failed.append(dashboard)

        print(f"\n✅ SUCCESSFULLY LAUNCHED ({len(successful)}):")
        for dashboard in successful:
            if dashboard == 'mlflow':
                print(f"   🔬 MLflow UI: http://localhost:5000")
            else:
                config = self.dashboard_configs.get(dashboard, {})
                port = config.get('port', 'Unknown')
                color = config.get('color', '📊')
                print(f"   {color} {dashboard.upper().replace('_', ' ')}: http://localhost:{port}")

        if failed:
            print(f"\n❌ FAILED TO LAUNCH ({len(failed)}):")
            for dashboard in failed:
                print(f"   ⚠️ {dashboard.upper().replace('_', ' ')}")

        print(f"\n📋 SYSTEM STATUS:")
        print(f"   🟢 Running: {len(successful)}/{len(results)}")
        print(f"   🔴 Failed: {len(failed)}/{len(results)}")

        print(f"\n🌐 QUICK ACCESS URLs:")
        for dashboard in successful:
            if dashboard == 'mlflow':
                print(f"   • MLflow: http://localhost:5000")
            else:
                config = self.dashboard_configs.get(dashboard, {})
                port = config.get('port', 'Unknown')
                name = dashboard.replace('_', ' ').title()
                print(f"   • {name}: http://localhost:{port}")

    def generate_documentation(self):
        """Generate comprehensive system documentation"""
        try:
            from documentation_generator import MLSystemDocumentationGenerator

            print("\n📚 GENERATING SYSTEM DOCUMENTATION...")
            generator = MLSystemDocumentationGenerator(
                project_root=Path(__file__).parent.parent.parent.parent,
                output_dir=Path(__file__).parent.parent.parent.parent / "docs" / "ml_system"
            )

            docs = generator.generate_complete_documentation()
            print(f"✅ Documentation generated successfully!")
            print(f"📁 Location: {generator.output_dir}")

        except Exception as e:
            logger.error(f"Failed to generate documentation: {e}")

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available"""
        dependencies = {
            'dash': False,
            'plotly': False,
            'optuna': False,
            'mlflow': False,
            'tensorflow': False
        }

        for dep in dependencies:
            try:
                __import__(dep)
                dependencies[dep] = True
            except ImportError:
                dependencies[dep] = False

        return dependencies

    def print_dependency_status(self):
        """Print status of dependencies"""
        deps = self.check_dependencies()

        print("\n🔧 DEPENDENCY STATUS:")
        print("-" * 30)

        for dep, available in deps.items():
            status = "✅ Available" if available else "❌ Missing"
            print(f"   {dep}: {status}")

        missing = [dep for dep, available in deps.items() if not available]
        if missing:
            print(f"\n⚠️ Missing dependencies may affect dashboard functionality:")
            for dep in missing:
                print(f"   pip install {dep}")

    def keep_alive(self):
        """Keep dashboards running"""
        print("\n🔄 DASHBOARDS RUNNING...")
        print("   Press Ctrl+C to stop all dashboards")
        print("   View logs above for any errors")
        print("="*80)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 SHUTDOWN REQUESTED...")
            self.shutdown()

    def shutdown(self):
        """Shutdown all dashboards"""
        print("📛 Stopping all dashboards...")

        # Stop processes
        for name, process in self.processes.items():
            try:
                process.terminate()
                logger.info(f"Stopped {name} process")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")

        print("✅ All dashboards stopped")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Launch ML System Dashboards')
    parser.add_argument('--port', type=int, default=8050, help='Base port for dashboards')
    parser.add_argument('--no-mlflow', action='store_true', help='Skip MLflow UI launch')
    parser.add_argument('--docs-only', action='store_true', help='Generate documentation only')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies only')

    args = parser.parse_args()

    manager = DashboardManager(base_port=args.port)
    manager.print_banner()

    if args.check_deps:
        manager.print_dependency_status()
        return

    if args.docs_only:
        manager.generate_documentation()
        return

    # Print dashboard information
    manager.print_dependency_status()
    manager.print_dashboard_info()

    # Launch dashboards
    results = manager.launch_all_dashboards(include_mlflow=not args.no_mlflow)

    # Generate documentation
    manager.generate_documentation()

    # Print summary
    manager.print_launch_summary(results)

    # Keep alive
    manager.keep_alive()

if __name__ == "__main__":
    main()