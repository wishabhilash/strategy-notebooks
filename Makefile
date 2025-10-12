.PHONY: create_layer

create_layer:
	@docker build --platform=linux/arm64 -t lambda-layer .
	@docker run --name lambda-layer lambda-layer
	@docker cp lambda-layer:/lambda ./layer
	@docker rm lambda-layer
