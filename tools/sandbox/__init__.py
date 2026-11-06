"""Local task, candidate, run and evidence management.

These helpers do not execute workloads. Mutations are serialized by the CLI's
workspace lock; callers using this package directly must acquire that lock.
"""
