import sys, os
sys.path.insert(0, "/root/.openclaw/workspace/titan")
os.chdir("/root/.openclaw/workspace/titan")
import pytest
sys.exit(pytest.main([
    "tests/test_engines.py",
    "-v", "--tb=short", "--no-header",
]))
