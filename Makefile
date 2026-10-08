.PHONY: install run quick report test app all
install:
	pip install -e ".[app,dev]"
run:
	python -m creditlab.pipeline
quick:
	python -m creditlab.pipeline --quick
report:
	python -m creditlab.report
test:
	pytest -q
app:
	streamlit run app/streamlit_app.py
all: run report test
