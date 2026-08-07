# Windows note: `make` is not installed by default. Each recipe below is a single
# command — run it directly in Git Bash or PowerShell if make is unavailable.

.PHONY: setup lock test dev ui docker offline-check clean

setup:
	python -m pip install -r requirements.txt

# Run this on the FIRST machine that installs cleanly, then commit the lock file.
# requirements.txt states intent; requirements.lock.txt is the reproducibility contract.
lock:
	python -m pip freeze > requirements.lock.txt

test:
	pytest

dev:
	uvicorn src.pipeline.api:app --reload --host 127.0.0.1 --port 8000

ui:
	streamlit run src/ui/app.py

docker:
	docker compose up --build

# CP3 gate + submission checklist: prove there is no network dependency.
offline-check:
	docker run --rm --network none deepguard-api python -c "import mediapipe, cv2, librosa; print('offline imports OK')"

clean:
	rm -rf .pytest_cache __pycache__ src/**/__pycache__ tests/__pycache__
