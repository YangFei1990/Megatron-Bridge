"""Canary file intentionally lacking a copyright header.

This file exists to verify that the repository's copyright-check workflow
detects Python files missing the standard NVIDIA copyright notice. If the
copyright-check job is working correctly, this file should cause the job
to fail.
"""


def canary():
    """Return a sentinel string for the copyright-check canary test."""
    return "missing-copyright-canary"
