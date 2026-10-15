# Rebuild figures and tables from the derived results in results/ (no trip data needed).
PYTHON ?= python

.PHONY: test figures tables all
all: test figures tables

test:
	$(PYTHON) -m pytest -q tests/

figures:
	mkdir -p results/figures
	for s in fig1_v3 fig2_v3 fig3_v3 fig4_v3 si_figures ed_figures ed_oneway_distance toy_tide sfig12_mirror; do $(PYTHON) src/$$s.py || exit 1; done

tables:
	$(PYTHON) src/table_s8.py
