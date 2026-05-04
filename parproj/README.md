# GraphPulse

GraphPulse is a hybrid CPU/GPU graph analytics pipeline with a small web dashboard. The backend is a C++17 CMake project that loads or generates graphs, extracts structural features, trains a Random Forest model, and reports runtime metrics. The frontend is an Express server that runs the backend binary and serves the dashboard.

## Project Layout

- `backend/` C++/CUDA source, headers, datasets, scripts, tests, and runtime plot assets used by the website
- `frontend/` Express server and static dashboard
- `Report/graphpulse.pdf` original project deliverable
- `Report/graphpulse_report.tex` / `Report/graphpulse_report.pdf` formal report (LaTeX + PDF)

## Prerequisites

- CMake 3.18+
- A C++17 compiler
- OpenMP-capable toolchain for parallel CPU execution
- CUDA Toolkit if you want GPU preprocessing
- Node.js for the web dashboard

## Build the Backend

CPU-only build:

```bash
cd backend
cmake -S . -B build -DENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CPU + GPU build:

```bash
cd backend
cmake -S . -B build_cuda -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda -j
```

## Run the Backend

Use a synthetic graph:

```bash
cd backend
./build/graph_ml --vertices 10000 --threads 8
```

Use a dataset file:

```bash
cd backend
./build/graph_ml --input data/ca-GrQc.txt --threads 8
```

Useful flags:

- `--vertices N`
- `--trees N`
- `--bc-samples N`
- `--threads N`
- `--sequential`
- `--gpu`
- `--input FILE`
- `--help`

## Run the Web App

```bash
cd frontend
npm install
npm start
```

Then open [http://localhost:3000](http://localhost:3000).

The dashboard reads benchmark CSVs and plot images from `backend/plots/presentation/`, so those assets are kept in the repository.

## Reports

- `Report/graphpulse.pdf`: initial project deliverable.
- `Report/graphpulse_report.pdf`: expanded report generated from `Report/graphpulse_report.tex`.

## Tests

```bash
cd backend
cmake -S . -B build -DENABLE_CUDA=OFF -DBUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```
