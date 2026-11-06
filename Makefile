PYTHON ?= python
export PYTHONPATH := src

.PHONY: all smoke verify test report app

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

app:
	$(PYTHON) -m streamlit run app/viewer.py --server.port 8501
