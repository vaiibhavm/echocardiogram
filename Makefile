PYTHON ?= python
CONFIG ?= configs/proposed_binary.yaml
DATA_ROOT ?= /path/to/CAMUS_public

install:
	$(PYTHON) -m pip install -e .

smoke:
	$(PYTHON) scripts/smoke_test.py

test:
	$(PYTHON) -m pytest

check-data:
	$(PYTHON) scripts/check_dataset.py --config $(CONFIG) --data-root "$(DATA_ROOT)"

train:
	$(PYTHON) scripts/train.py --config $(CONFIG) --data-root "$(DATA_ROOT)"
