---
type: rule
name: progression
---
{"predicate": "threshold", "args": {"metric": "bench_weight", "op": ">", "value": 65}, "on_pass": "increase weight by 2.5kg", "on_fail": "maintain current weight"}
