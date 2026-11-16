.PHONY: setup test data analyze eval calibrate
setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt && cp -n .env.example .env || true
test:
	python -m pytest tests -q
data:
	python data/build_leases.py
analyze:
	python -m fineprint analyze data/leases/lease_a.txt
eval:
	python -m evals.run_evals --grid default
calibrate:
	python -m evals.calibrate_judge --repeats 3
