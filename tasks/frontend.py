"""
Agent Demo Webpage

Loads frontend HTML from file — avoids Python string escaping issues.
"""

import os

_dir = os.path.dirname(os.path.abspath(__file__))
_html_path = os.path.join(_dir, "frontend.html")

with open(_html_path, "r", encoding="utf-8") as f:
    HTML = f.read()