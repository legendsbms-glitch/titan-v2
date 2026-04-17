# FILE: main.py
"""
TITAN v2.0 — Entry Point

Usage:
  python main.py analyze          → Run full 9-engine analysis
  python main.py analyze --force  → Force re-run (bypass cache)
  python main.py analyze --json   → Output as JSON
  python main.py api              → Start FastAPI server
  python main.py api --port 8000
  python main.py dashboard        → Streamlit dashboard
  python main.py schedule         → Auto-run every 15 minutes
  python main.py schedule --interval 30
  python main.py backtest --start 2023-01-01 --end 2024-01-01 --capital 10000
  python main.py test             → Run pytest suite
"""
import sys
import os
import argparse

# Ensure titan root is in path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║  ████████╗██╗████████╗ █████╗ ███╗   ██╗                ║
║     ██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║                ║
║     ██║   ██║   ██║   ███████║██╔██╗ ██║                ║
║     ██║   ██║   ██║   ██╔══██║██║╚██╗██║                ║
║     ██║   ██║   ██║   ██║  ██║██║ ╚████║                ║
║     ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝                ║
║  v2.0 — 9-Engine Gold Intelligence System (FREE STACK)  ║
╚══════════════════════════════════════════════════════════╝
"""


def cmd_analyze(args):
    """Run full TITAN analysis"""
    from core.db import init_db
    from core.titan_runner import titan_analyze
    init_db()
    verdict = titan_analyze(verbose=not args.json, force=args.force)
    if args.json:
        import json
        # Remove bulky engine_results for clean output unless --full
        output = {k: v for k, v in verdict.items() if k not in ("engine_results",)}
        print(json.dumps(output, indent=2, default=str))
    return verdict


def cmd_api(args):
    """Start FastAPI server"""
    import uvicorn
    from core.db import init_db
    init_db()
    print(f"\n⚡ TITAN API starting on http://{args.host}:{args.port}")
    print(f"   Docs: http://{args.host}:{args.port}/docs\n")

    # Import here to avoid circular at top level
    sys.path.insert(0, ROOT)
    from api.routes import app
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


def cmd_dashboard(args):
    """Launch Streamlit dashboard"""
    import subprocess
    dashboard_path = os.path.join(ROOT, "dashboard", "app.py")
    print(f"\n⚡ TITAN Dashboard starting on http://localhost:{args.port}\n")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard_path,
        f"--server.port={args.port}",
        "--server.headless=true",
    ])


def cmd_schedule(args):
    """Run scheduled analysis"""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from core.db import init_db
    from core.titan_runner import titan_analyze
    from alerts.alert_engine import run as run_alerts

    init_db()
    scheduler = BlockingScheduler()

    print(f"\n⚡ TITAN Scheduler: running every {args.interval} minutes\n")

    @scheduler.scheduled_job("interval", minutes=args.interval)
    def scheduled_job():
        print(f"\n[{__import__('datetime').datetime.utcnow().strftime('%H:%M:%S')}] Running TITAN analysis...")
        try:
            verdict = titan_analyze(verbose=True, force=True)
            # Check alerts
            alert_result = run_alerts(verdict=verdict)
            if alert_result.get("alert_type"):
                print(f"🚨 Alert: {alert_result.get('alert_type')}")
        except Exception as e:
            print(f"Scheduled job error: {e}")

    # Run once immediately
    scheduled_job()
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\nScheduler stopped.")


def cmd_backtest(args):
    """Run historical backtest"""
    from backtesting.backtester import run_backtest
    print(f"\n⚡ TITAN Backtest: {args.start} → {args.end} | Capital: ${args.capital:,.0f}\n")
    result = run_backtest(
        start=args.start, end=args.end,
        capital=args.capital, risk_pct=args.risk,
    )

    print(f"\n{'='*50}")
    print(f"TITAN BACKTEST — {args.start} to {args.end}")
    print(f"{'='*50}")
    print(f"Trades:        {result.total_trades}")
    print(f"Win Rate:      {result.win_rate:.1%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Expectancy:    ${result.expectancy:.2f}/trade")
    print(f"Sharpe Ratio:  {result.sharpe:.2f}")
    print(f"Max Drawdown:  ${result.max_drawdown:.2f}")
    print(f"Total Return:  ${result.total_return:.2f} ({result.total_pct:.1f}%)")
    mc = result.monte_carlo
    if "final_equity" in mc:
        fe = mc["final_equity"]
        print(f"\nMonte Carlo (1000 sims):")
        print(f"  P5:  ${fe['p5']:,.0f}  |  P50: ${fe['p50']:,.0f}  |  P95: ${fe['p95']:,.0f}")
        print(f"  Probability of profit: {mc.get('probability_of_profit',0):.1%}")
    print(f"{'='*50}\n")


def cmd_test(args):
    """Run pytest test suite"""
    import subprocess
    test_path = os.path.join(ROOT, "tests", "test_engines.py")
    result = subprocess.run([
        sys.executable, "-m", "pytest", test_path,
        "-v", "--tb=short", "-x" if args.stop_first else ""
    ])
    sys.exit(result.returncode)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="TITAN v2.0 — 9-Engine Gold Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Run full analysis")
    p_analyze.add_argument("--verbose", "-v", action="store_true", default=True)
    p_analyze.add_argument("--force",   "-f", action="store_true", help="Bypass cache")
    p_analyze.add_argument("--json",    "-j", action="store_true", help="JSON output")

    # api
    p_api = subparsers.add_parser("api", help="Start FastAPI server")
    p_api.add_argument("--host", default="0.0.0.0")
    p_api.add_argument("--port", default=8000, type=int)

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    p_dash.add_argument("--port", default=8501, type=int)

    # schedule
    p_sched = subparsers.add_parser("schedule", help="Run on schedule")
    p_sched.add_argument("--interval", default=15, type=int, help="Minutes between runs")

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Historical backtest")
    p_bt.add_argument("--start",   default="2023-01-01", help="Start date YYYY-MM-DD")
    p_bt.add_argument("--end",     default="2024-01-01", help="End date YYYY-MM-DD")
    p_bt.add_argument("--capital", default=10000.0, type=float, help="Starting capital")
    p_bt.add_argument("--risk",    default=1.0, type=float, help="Risk per trade %%")

    # test
    p_test = subparsers.add_parser("test", help="Run test suite")
    p_test.add_argument("--stop-first", "-x", action="store_true")

    args = parser.parse_args()

    if not args.command or args.command == "analyze":
        if not hasattr(args, "force"):
            args.force   = False
            args.json    = False
            args.verbose = True
        cmd_analyze(args)

    elif args.command == "api":
        cmd_api(args)

    elif args.command == "dashboard":
        cmd_dashboard(args)

    elif args.command == "schedule":
        cmd_schedule(args)

    elif args.command == "backtest":
        cmd_backtest(args)

    elif args.command == "test":
        cmd_test(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
