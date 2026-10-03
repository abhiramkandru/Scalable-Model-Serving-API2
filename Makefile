VENV   ?= .venv
PY      = $(VENV)/bin/python
PIP     = $(VENV)/bin/pip

.PHONY: install test run bench load dashboard-install dashboard-dev dashboard-build up down rules grafana-dashboard

install:                       ## Python deps (server + tests)
	python3 -m venv $(VENV)
	$(PIP) install -r server/requirements-dev.txt

test:                          ## unit + API tests
	cd server && ../$(PY) -m pytest -q

run:                           ## API on :8000 with the simulated GPU
	cd server && ../$(VENV)/bin/uvicorn app.main:app --port 8000 --reload

bench:                         ## in-process scheduler benchmark (writes bench/results.json)
	cd server && ../$(PY) -m app.bench all --duration 8 --json ../bench/results.json

load:                          ## HTTP load against a running API
	$(PY) loadtest/loadtest.py --url http://localhost:8000 --rps 60 --duration 60

dashboard-install:
	cd dashboard && npm install

dashboard-dev:                 ## dashboard on :5173, proxying /api to :8000
	cd dashboard && npm run dev

dashboard-build:
	cd dashboard && npm run build

up:                            ## API + dashboard + Prometheus + Grafana
	docker compose -f deploy/docker/docker-compose.yml up --build

down:
	docker compose -f deploy/docker/docker-compose.yml down

rules:                         ## regenerate the GKE rules from deploy/prometheus/rules.yml
	$(PY) deploy/sync_rules.py

grafana-dashboard:             ## regenerate deploy/grafana/dashboards/inference.json
	$(PY) deploy/grafana/gen_dashboard.py
