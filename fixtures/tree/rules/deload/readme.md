---
type: rule
name: deload
---
{"predicate": "threshold", "args": {"metric": "ohp_weight", "op": "<", "value": 40}, "on_pass": "deload to 80% for one session", "on_fail": "continue"}
