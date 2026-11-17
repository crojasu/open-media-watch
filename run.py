#!/usr/bin/env python3
"""run.py — local development entry point.

Usage:
  python run.py              # http://127.0.0.1:8000 (auto-reload)
  python run.py --port 8080
  python run.py --prod       # no auto-reload
"""
import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open Media Watch API")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--prod", action="store_true", help="no auto-reload")
    args = parser.parse_args()
    uvicorn.run("fmedia.main:app", host=args.host, port=args.port, reload=not args.prod)
