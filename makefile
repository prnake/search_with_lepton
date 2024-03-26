.PHONY: build run

# Default command to build and run the server
all: build run

# Install dependencies
build:
	cd web && pnpm install && pnpm run build

# Run the server with environment variables
run:
	uvicorn search:app --workers 4 --port 8080

test:
	uvicorn search:app --reload --port 8080

# Helper command to clean the project
clean:
	rm -rf ui