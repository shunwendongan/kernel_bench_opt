"""计时计划必须覆盖全部样本，且每轮恰好一次。"""
import importlib.util
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_schedule_balances_rounds_without_dropping_samples():
    assert importlib.util.find_spec('run_board') is not None, '尚未实现批量采集'
    from run_board import measurement_schedule
    samples = [{'id': n} for n in range(40)]
    schedule = measurement_schedule(samples, 10, 42)
    assert len(schedule) == 400
    assert schedule == measurement_schedule(samples, 10, 42)
    assert Counter(s['id'] for _, s in schedule) == {n: 10 for n in range(40)}
    for repeat in range(10):
        assert {s['id'] for r, s in schedule if r == repeat} == set(range(40))
    assert [s['id'] for r, s in schedule if r == 0] != [s['id'] for r, s in schedule if r == 1]
