"""Backward-compatibility module — shadowed by the gui/ package.

Python always loads the gui/ directory package in preference to this file.
This module is kept only so that tools or IDEs that inspect the filesystem
do not see a completely empty location.  Do not add logic here.
"""
# This file is intentionally a no-op.
# All symbols live in src/interfaces/gui/ (package).
