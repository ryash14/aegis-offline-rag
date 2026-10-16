.PHONY: test check run doctor

test:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q
check:
	uvx --from ruff==0.17.1 ruff check aegis scripts tests app.py --select F
	uvx --from ruff==0.17.1 ruff format --check aegis scripts tests app.py
	.venv/bin/python -m compileall -q aegis scripts tests app.py
run:
	./scripts/run-local.sh
doctor:
	.venv/bin/python -m aegis.cli --db data/squad.sqlite doctor
