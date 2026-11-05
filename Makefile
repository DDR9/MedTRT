PYTHON ?= ../medical-ai/.venv/Scripts/python.exe
PYTHONPATH := src

.PHONY: env data bench eval report app test all

env:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medtrt.config --env-audit

data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medtrt.data

bench:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medtrt.bench --backend $(BACKEND)

eval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medtrt.evaluate --backend $(BACKEND)

report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m medtrt.report

app:
	PYTHONPATH=$(PYTHONPATH) ../medical-ai/.venv/Scripts/streamlit.exe run app/viewer.py --server.port 8501

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/

all: env data bench eval report
