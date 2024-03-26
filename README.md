
## Install requirements

```shell
pip install -r requirements.txt
```

## Run

### 1. Build the front-end web

```shell
cd web && pnpm install && pnpm run build
```
Output: The project root directory produces a `ui` folder, which contains static files for the frontend.

### 2. Run server with Lepton API

set `OPENAI_BASE_URL`, `OPENAI_TOKEN`, `SERPER_API` before run server

```shell
uvicorn search:app --workers 4 --port 8080
```
ok, now your search app running on http://0.0.0.0:8081

## Reference

- https://github.com/leptonai/search_with_lepton
- https://github.com/shibing624/SmartSearch
- https://github.com/Neutrino1998/search_with_langchain