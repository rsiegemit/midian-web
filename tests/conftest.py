import os, tempfile
os.environ.setdefault("RTE_LLM_CACHE", tempfile.mkdtemp(prefix="rte_test_memo_"))   # never load the production memo (GBs) in tests
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
