.PHONY: run-kfp init-kfp-env

run-kfp:
	./run_kfp_submit_and_watch.sh

init-kfp-env:
	@if [ ! -f .env.kfp ]; then \
		cp .env.kfp.example .env.kfp; \
		echo "Created .env.kfp from .env.kfp.example"; \
	else \
		echo ".env.kfp already exists"; \
	fi
