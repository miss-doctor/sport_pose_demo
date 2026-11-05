#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""摄像头实时姿态推理（video_demo.py 的便捷入口）。"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

cmd = [sys.executable, os.path.join(ROOT, "demo", "video_demo.py"),
       "--input", "0", "--show"] + sys.argv[1:]
subprocess.run(cmd, check=False)
