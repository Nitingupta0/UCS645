import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;
const projectRoot = path.resolve(__dirname, '..', 'backend');
const cpuBinaryPath = path.join(projectRoot, 'build', 'graph_ml');
const gpuBinaryPath = path.join(projectRoot, 'build_cuda', 'graph_ml');
const dataDir = path.join(projectRoot, 'data');
const presentationDir = path.join(projectRoot, 'plots', 'presentation');

app.use(express.static(path.join(__dirname, 'public')));
app.use('/assets/plots', express.static(presentationDir));
app.use(express.json());
app.use(cors());

app.get('/favicon.ico', (req, res) => {
    res.setHeader('Content-Type', 'image/svg+xml');
    res.send(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
        + "<rect width='64' height='64' rx='12' fill='#0c0f0f'/>"
        + "<circle cx='20' cy='32' r='6' fill='#A5B4FC'/>"
        + "<circle cx='44' cy='22' r='6' fill='#FBBF24'/>"
        + "<circle cx='44' cy='42' r='6' fill='#FBBF24'/>"
        + "<path d='M24 32 L38 24 M24 32 L38 40' stroke='#c4c7c7' stroke-width='3'/>"
        + "</svg>"
    );
});

function readCsvFile(filePath) {
    if (!fs.existsSync(filePath)) {
        return [];
    }

    const raw = fs.readFileSync(filePath, 'utf-8').trim();
    if (!raw) {
        return [];
    }

    const lines = raw.split(/\r?\n/);
    const headers = lines.shift().split(',');

    return lines
        .filter((line) => line.trim())
        .map((line) => {
            const values = line.split(',');
            const row = {};

            headers.forEach((key, idx) => {
                const value = values[idx] ?? '';
                const num = Number(value);
                row[key] = Number.isFinite(num) ? num : value;
            });

            return row;
        });
}

function listDatasets() {
    if (!fs.existsSync(dataDir)) {
        return [];
    }

    return fs.readdirSync(dataDir)
        .filter((file) => file.endsWith('.txt'))
        .map((file) => ({
            name: file.replace(/\.txt$/i, ''),
            file
        }));
}

function parseResultMetrics(output) {
    const resultLine = output.match(/^\[RESULT\]\s+(.+)$/m);
    if (!resultLine) {
        return null;
    }

    const metrics = {};
    for (const pair of resultLine[1].split(',')) {
        const [key, rawValue] = pair.split('=');
        if (!key || rawValue === undefined) {
            continue;
        }

        const value = Number(rawValue.trim());
        metrics[key.trim()] = Number.isFinite(value) ? value : rawValue.trim();
    }

    return metrics;
}

app.get('/api/datasets', (req, res) => {
    res.json({ datasets: listDatasets() });
});

app.get('/api/presentation/datasets', (req, res) => {
    const datasetFile = path.join(presentationDir, 'dataset_results.csv');
    res.json({ rows: readCsvFile(datasetFile) });
});

app.get('/api/presentation/perf', (req, res) => {
    const mode = req.query.mode === 'gpu' ? 'gpu' : 'cpu';
    const filePath = path.join(presentationDir, `perf_summary_${mode}.csv`);
    res.json({ rows: readCsvFile(filePath) });
});

app.post('/api/execute', (req, res) => {
    const { vertices = 5000, threads = 8, gpu = false, dataset = '' } = req.body;
    const v = Number.parseInt(vertices, 10);
    const t = Number.parseInt(threads, 10);

    if (!Number.isInteger(v) || !Number.isInteger(t)) {
        return res.status(400).json({ error: 'vertices and threads must be integers' });
    }

    const safeVertices = Math.min(Math.max(v, 100), 1000000);
    const safeThreads = Math.min(Math.max(t, 1), 256);
    const wantsGpu = gpu === true || gpu === 'true' || gpu === 1;
    const useGpu = wantsGpu && fs.existsSync(gpuBinaryPath);
    const selectedBinary = useGpu ? gpuBinaryPath : cpuBinaryPath;

    if (!fs.existsSync(selectedBinary)) {
        return res.status(500).json({
            error: 'graph_ml binary not found',
            hint: 'Build backend first with CMake so build/graph_ml or build_cuda/graph_ml exists'
        });
    }

    const args = [];
    const safeDataset = typeof dataset === 'string' ? path.basename(dataset) : '';

    if (safeDataset) {
        const datasetPath = path.join(dataDir, safeDataset);
        if (!fs.existsSync(datasetPath)) {
            return res.status(400).json({ error: 'dataset not found' });
        }
        args.push('--input', datasetPath);
    } else {
        args.push('--vertices', String(safeVertices));
    }

    args.push('--threads', String(safeThreads));
    if (useGpu) {
        args.push('--gpu');
    }

    let child;
    try {
        child = spawn(selectedBinary, args, { cwd: projectRoot });
    } catch (error) {
        console.error(`Error executing graph_ml: ${error.message}`);
        return res.status(500).json({ error: error.message });
    }

    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('X-Accel-Buffering', 'no');

    let output = '';
    let stderrOutput = '';

    child.stdout.on('data', (chunk) => {
        const text = chunk.toString();
        output += text;
        res.write(text);
    });

    child.stderr.on('data', (chunk) => {
        const text = chunk.toString();
        stderrOutput += text;
        res.write(text);
    });

    child.on('error', (error) => {
        console.error(`Error executing graph_ml: ${error.message}`);
        const payload = JSON.stringify({
            success: false,
            error: error.message,
            stderr: stderrOutput
        });
        res.write(`\n__GRAPH_PULSE_RESULT__${payload}\n`);
        res.end();
    });

    child.on('close', (code) => {
        if (code !== 0) {
            const payload = JSON.stringify({
                success: false,
                error: `graph_ml exited with code ${code}`,
                stderr: stderrOutput
            });
            res.write(`\n__GRAPH_PULSE_RESULT__${payload}\n`);
            res.end();
            return;
        }

        let accuracy = 'N/A';
        let cpuVertices = '0';
        let gpuVertices = '0';
        let throughput = 'N/A';
        let speedup = 'N/A';
        let edges = 'N/A';
        let totalMs = 'N/A';

        const accMatch = output.match(/Accuracy:\s+([\d.]+)\s*%/);
        if (accMatch) {
            accuracy = accMatch[1];
        }

        const splitMatch = output.match(/CPU vertices:\s+(\d+)\s+GPU vertices:\s+(\d+)/);
        if (splitMatch) {
            cpuVertices = splitMatch[1];
            gpuVertices = splitMatch[2];
        }

        const resultMetrics = parseResultMetrics(output);
        if (resultMetrics) {
            edges = Number(resultMetrics.edges);
            totalMs = Number(resultMetrics.total_ms);
            const threadsUsed = Number(resultMetrics.threads);

            if (Number.isFinite(edges) && Number.isFinite(totalMs) && totalMs > 0) {
                throughput = (edges / totalMs).toFixed(2);
            }

            if (Number.isFinite(threadsUsed) && threadsUsed > 0) {
                speedup = threadsUsed.toFixed(0);
            }
        }

        const payload = JSON.stringify({
            success: true,
            metrics: {
                accuracy,
                cpuVertices,
                gpuVertices,
                throughput,
                speedup,
                edges,
                totalMs,
                engine: useGpu ? 'gpu' : 'cpu'
            }
        });

        res.write(`\n__GRAPH_PULSE_RESULT__${payload}\n`);
        res.end();
    });
});

app.listen(PORT, () => {
    console.log(`GraphPulse Web Server running on http://localhost:${PORT}`);
});
