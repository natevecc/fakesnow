import sys
import urllib.request

try:
    response = urllib.request.urlopen("http://localhost:8000/health", timeout=2)
    sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
