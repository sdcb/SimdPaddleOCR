import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const wwwroot = path.join(__dirname, 'bin', 'Release', 'net10.0', 'publish', 'wwwroot');
const datasetDir = path.join(repoRoot, 'dataset');

// Parse CLI flags
const argv = process.argv.slice(2);
function getArg(name, def) {
    const idx = argv.indexOf(name);
    return (idx !== -1 && idx + 1 < argv.length) ? argv[idx + 1] : def;
}
const workers = getArg('--workers', '4');
const model = getArg('--model', 'tiny');
const count = getArg('--count', '0');
const out = getArg('--out', `bench-out/wasm-${model}-${workers}w.json`);
const port = parseInt(getArg('--port', '0'), 10);
const isHeadless = !argv.includes('--open');
const noLaunch = argv.includes('--no-launch');
// --env KEY=VALUE (repeatable): environment variables for the .NET runtime.
const envPairs = argv.flatMap((a, i) => a === '--env' && i + 1 < argv.length ? [argv[i + 1]] : []);

const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.mjs': 'application/javascript',
    '.wasm': 'application/wasm',
    '.json': 'application/json',
    '.dat': 'application/octet-stream',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.css': 'text/css',
};

const server = http.createServer((req, res) => {
    // Crucial for SharedArrayBuffer / Web Workers multi-threading
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Access-Control-Allow-Origin', '*');

    const url = new URL(req.url, 'http://localhost');

    // 1. API: list dataset files
    if (url.pathname === '/api/dataset-files') {
        if (fs.existsSync(datasetDir)) {
            const files = fs.readdirSync(datasetDir)
                .filter(f => f.toLowerCase().endsWith('.jpg') || f.toLowerCase().endsWith('.png'))
                .sort();
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(files));
        } else {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify([]));
        }
        return;
    }

    // 2. API: log message from browser
    if (url.pathname === '/api/log') {
        let body = '';
        req.on('data', chunk => body += chunk);
        req.on('end', () => {
            if (body) console.log(body);
            res.writeHead(200);
            res.end();
        });
        return;
    }

    // 2. API: save benchmark result
    if (url.pathname === '/api/save-benchmark') {
        const outFileName = url.searchParams.get('out') || out;
        const targetPath = path.isAbsolute(outFileName) ? outFileName : path.join(repoRoot, outFileName);

        let body = '';
        req.on('data', chunk => body += chunk);
        req.on('end', () => {
            try {
                fs.mkdirSync(path.dirname(targetPath), { recursive: true });
                fs.writeFileSync(targetPath, body, 'utf8');
                console.log(`\n[SERVER] Saved benchmark result to: ${targetPath}`);
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ ok: true, path: targetPath }));
            } catch (err) {
                console.error('[SERVER] Failed to save benchmark:', err);
                res.writeHead(500, { 'Content-Type': 'text/plain' });
                res.end('Failed: ' + err.message);
            }
        });
        return;
    }

    // 3. API: execution done
    if (url.pathname === '/api/done') {
        const exitCode = parseInt(url.searchParams.get('exitCode') || '0', 10);
        const err = url.searchParams.get('error');
        if (err) {
            console.error('\n[SERVER] Browser reported error:\n', decodeURIComponent(err));
        }
        console.log(`\n[SERVER] Benchmark run completed. Exit code: ${exitCode}`);
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('OK');

        setTimeout(() => {
            server.close();
            process.exit(exitCode);
        }, 500);
        return;
    }

    // 4. Static files from /dataset/
    if (url.pathname.startsWith('/dataset/')) {
        const rel = url.pathname.substring('/dataset/'.length);
        const filePath = path.join(datasetDir, rel);
        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            const ext = path.extname(filePath).toLowerCase();
            res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
            fs.createReadStream(filePath).pipe(res);
            return;
        } else {
            res.writeHead(404);
            res.end('Dataset file not found: ' + rel);
            return;
        }
    }

    // 5. Static files from wwwroot
    let filePath = path.join(wwwroot, url.pathname === '/' ? 'index.html' : url.pathname);
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
        res.writeHead(404);
        res.end('File not found: ' + url.pathname);
        return;
    }

    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
    fs.createReadStream(filePath).pipe(res);
});

server.listen(port, '127.0.0.1', () => {
    const assignedPort = server.address().port;
    const queryParams = new URLSearchParams({
        workers,
        model,
        count,
        out,
        autorun: '1'
    });
    for (const pair of envPairs) queryParams.append('env', pair);
    // 把方向验证开关传入页面，再由页面转为 .NET 的 --orientation 参数。
    if (argv.includes('--orientation')) queryParams.set('orientation', '1');
    const targetUrl = `http://127.0.0.1:${assignedPort}/index.html?${queryParams.toString()}`;

    console.log(`[SERVER] WasmBench server running at http://127.0.0.1:${assignedPort}/`);
    console.log(`[SERVER] Target URL: ${targetUrl}`);

    if (noLaunch) {
        console.log(`[SERVER] --no-launch specified. Open the target URL in your browser to run.`);
        return;
    }

    const edgeCandidates = [
        "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "msedge",
        "chrome"
    ];

    let browserPath = edgeCandidates.find(p => fs.existsSync(p));
    if (!browserPath) browserPath = "msedge";

    // 使用工作区独立的临时配置目录，避免复用桌面 Edge 进程或写入只读用户配置。
    const browserScratch = path.join(repoRoot, '.tmp-artifacts');
    fs.mkdirSync(browserScratch, { recursive: true });
    const browserProfile = fs.mkdtempSync(path.join(browserScratch, 'wasm-browser-'));

    const browserArgs = isHeadless
        ? ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--log-level=3', `--user-data-dir=${browserProfile}`, targetUrl]
        : [targetUrl];
    // 仅显式请求时关闭测试浏览器沙箱，用于受限环境里的本机可信页面。
    if (argv.includes('--browser-no-sandbox')) browserArgs.push('--no-sandbox');

    console.log(`[SERVER] Launching ${path.basename(browserPath)} (${isHeadless ? 'Headless' : 'Windowed'})...`);
    const browser = spawn(browserPath, browserArgs, { windowsHide: isHeadless, stdio: isHeadless ? ['ignore', 'ignore', 'pipe'] : 'inherit' });
    browser.stderr?.on('data', chunk => process.stderr.write(chunk));

    browser.on('error', (err) => {
        console.error('[SERVER] Failed to launch browser:', err.message);
        console.log(`[SERVER] Please manually open: ${targetUrl}`);
    });
});
