APP_NAME=tweak-story
DOCKER_CMD=$(shell which docker || which podman || echo "docker")
PORT=8501
MODEL_CHECKPOINT_ID=1EwzENhOilz9bKLIVcz22qn0Ip8wIwwFr
MODEL_CHECKPOINT_NAME=BEST_checkpoint_flickr8k_1_cap_per_img_1_min_word_freq.pth
WORD_MAP_CHECKPOINT_ID=1FW4J2ZB3BQd_7zdxxgjZ63WMbFYm2crn
WORD_MAP_CHECKPOINT_NAME=WORDMAP_flickr8k_1_cap_per_img_1_min_word_freq.json
CONFIG_CHECKPOINT_ID=1yTtOwbFVDPUE7Jpnlq9aG5y0_jEgd9rS

.PHONY: get_checkpoints run stop
get_checkpoints:
	@echo unimplemented! Download checkpoints manually as described here: https://github.com/namanphy/Controllable-Image-Captioning-App
build:
	$(DOCKER_CMD) build -t $(APP_NAME) .

run: build
	$(DOCKER_CMD) run -d -p=$(PORT):$(PORT) --rm --name=$(APP_NAME) $(APP_NAME)
	@echo $(APP_NAME) running at localhost:$(PORT)

stop:
	$(DOCKER_CMD) stop $(APP_NAME) || true
