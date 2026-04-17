import sys
sys.path.insert(0, "/root/.openclaw/workspace/titan")
from core.db import init_db
init_db()
from engines.engine1_price_matrix import run
import json
result = run()
print(json.dumps(result, indent=2, default=str))
