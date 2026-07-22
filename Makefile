# Set to a specific GPU index (e.g. "1") to restrict to that GPU,
# or leave empty to use all available GPUs.
CUDA_DEVICE ?=

.PHONY: test lint format build run

train-rayleigh:
	cd scripts && \
	uv run python train.py \
	--epochs 150 \
	--distribution_params '{"scale": 1.0}' \
	--distribution rayleigh \
	--gan_type wgan_gp \
	--num_workers 15 \
	--num_test_samples 2000 \
	--batch_size \
	-k 5

train-nakagami:
	cd scripts && \
	uv run python train.py \
	--epochs 100 \
	--distribution_params '{"nu": 3.1415, "scale": 10}' \
	--distribution nakagami \
	--num_workers 15 \
	--gan_type gan \
	--num_test_samples 5000 \
	--batch_size 256

train-mftr:
	cd scripts && \
	uv run python train.py \
	--epochs 160 \
	--distribution_params '{"m": 8, "mu": 3, "K": 8.0, "delta": 0.9, "omega": 2.0}' \
	--distribution mftr \
	--num_workers 15 \
	--gan_type gan \
	--num_test_samples 2000 \
	--batch_size 256 \
	-d 1028 \
	-k 5 \
	--betas_g 0.6 0.999 \
	--num_samples 20000

train-cgan-mftr:
	cd scripts && \
	uv run python train_cgan.py \
	--epochs 20 \
	--batch_size 1024 \
	--latent_dim 1024 \
	--g_every_k_steps 3 \
	--cond_dim 5 \
	--cond_emb_dim 32 \
	--samples_per_combo 1000 \
	--val_samples_per_combo 1000 \
	--combos 100000 \
	--val_combos 128 \
	--m_range "[1, 8]" \
	--mu_range "[1, 9]" \
	--K_range "[0, 20]" \
	--delta_range "[0, 0.9]" \
	--omega_range "[1, 2]" \
	--num_workers 12 \
	--lr_g 0.0002 \
	--lr_d 0.0004 \
	--betas_g 0.5 0.999

train-cgan-scenarios:
	cd scripts && \
	$(if $(CUDA_DEVICE),CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) )uv run python train_cgan_scenarios.py \
	--train_scenarios 1 2 3 4 5 \
	--source empirical_pdf \
	--samples_per_scenario 10000 \
	--epochs 150 \
	--batch_size 256 \
	--latent_dim 1024 \
	--cond_emb_dim 32 \
	--g_every_k_steps 5 \
	--num_workers 12 \
	--lr_g 0.0002 \
	--lr_d 0.0004 \
	--betas_g 0.5 0.999 \
	--devices 1 \
	--normalize_conds \
	--seed 42

test-gan:
	cd scripts && \
	uv run python test.py \
	--distribution nakagami \
	--num_test_samples 2000 \
	--num_workers 15 \
	--gan_type gan \
	-v 4

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts

build:
	uv build

install-hooks:
	uv run pre-commit install

run-hooks:
	uv run pre-commit run --all-files

tensorboard:
	tensorboard --logdir logs/

clean:
	rm -rf logs/* || true && rm -rf results/tests/* || true

run:
	cd src/app && \
	uv run python main.py --share

dev:
	cd src/app && \
	uv run gradio main.py
