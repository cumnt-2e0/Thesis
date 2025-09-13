.PHONY: setup train eval exp test
setup:
	python -m pip install -e .
train:
	mgrl-train --env_id ieee33 --total_timesteps 150000
eval:
	mgrl-eval --env_id ieee33 --model_path artifacts/models/ppo_ieee33 --episodes 5
exp:
	mgrl-exp --timesteps 100000 --episodes 3 --out artifacts/runs/summary.csv
test:
	pytest -q
