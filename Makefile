PYTHON ?= python
export PYTHONPATH := src

.PHONY: all smoke verify test report paper app

all:
	$(PYTHON) -m medtrt.pipeline

smoke:
	$(PYTHON) -m medtrt.pipeline --smoke

verify:
	$(PYTHON) -m medtrt.verify_claims

test:
	$(PYTHON) -m pytest tests/

report:
	$(PYTHON) -m medtrt.report

paper:
	cd docs && pdflatex -interaction=nonstopmode -halt-on-error TECHNICAL_PAPER.tex && pdflatex -interaction=nonstopmode -halt-on-error TECHNICAL_PAPER.tex
	cp docs/TECHNICAL_PAPER.pdf MedTRT_Scientific_Report.pdf
	rm -f docs/TECHNICAL_PAPER.aux docs/TECHNICAL_PAPER.log docs/TECHNICAL_PAPER.out

app:
	$(PYTHON) -m streamlit run app/viewer.py --server.port 8501
