"""
Beyond the Scoreboard: Cricket Franchise Decision Intelligence System
Phase 1 Environment & Data Readiness Verification Script
"""

import sys
import os
import json
import urllib.request
import zipfile
import io

def run_environment_check():
    print("=" * 70)
    print("CRICKET DECISION INTELLIGENCE SYSTEM: ENVIRONMENT CHECK")
    print("=" * 70)
    
    # 1. Check Python version
    py_version = sys.version.split()[0]
    print(f"[+] Python Version: {py_version}")
    major, minor = sys.version_info.major, sys.version_info.minor
    if major < 3 or (major == 3 and minor < 10):
        print(f"[-] WARNING: Python >= 3.10 is recommended. Found: {py_version}")
    else:
        print("[+] Python version check PASSED.")

    # 2. Check Directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    required_dirs = [
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "sql"),
        os.path.join(base_dir, "src", "data_pipeline"),
        os.path.join(base_dir, "src", "analytics"),
        os.path.join(base_dir, "src", "models"),
        os.path.join(base_dir, "notebooks"),
        os.path.join(base_dir, "powerbi"),
        os.path.join(base_dir, "reports")
    ]
    
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    if missing_dirs:
        print(f"[-] Missing directories: {missing_dirs}")
    else:
        print(f"[+] All {len(required_dirs)} project directories verified.")

    # 3. Check Cricsheet Endpoint Connectivity
    url = "https://cricsheet.org/downloads/ipl_csv2.zip"
    print(f"\n[+] Testing connection to Cricsheet endpoint: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CricketAnalytics/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_len = resp.headers.get("Content-Length", 0)
            status_code = resp.status
            size_mb = round(int(content_len) / (1024 * 1024), 2)
            print(f"[+] Remote HTTP Status: {status_code} OK")
            print(f"[+] Remote Zip Archive Size: {size_mb} MB")
            print("[+] Cricsheet data connectivity check PASSED.")
    except Exception as e:
        print(f"[-] ERROR connecting to Cricsheet: {e}")

    # 4. Check Core Libraries
    print("\n[+] Checking core Python dependencies:")
    deps = ["pandas", "numpy", "matplotlib", "scipy", "sklearn", "sqlite3"]
    for dep in deps:
        try:
            mod = __import__(dep)
            ver = getattr(mod, "__version__", "built-in")
            print(f"    - {dep:<12}: Available ({ver})")
        except ImportError:
            print(f"    - {dep:<12}: Not installed (will be installed from requirements.txt)")

    print("\n" + "=" * 70)
    print("PHASE 1 ENVIRONMENT VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_environment_check()
