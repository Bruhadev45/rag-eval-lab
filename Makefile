.PHONY: reproduce test clean

# Recompute every table, figure, and results.json from data/runs/.
reproduce:
	python -m rageval.report

test:
	pytest -q

# Remove derived outputs; `make reproduce` recreates them identically.
clean:
	rm -rf results/results.json results/TABLES.md results/figures/*.png
