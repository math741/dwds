// The MLG in a browser (web/build.sh), driven through the HUD's own input in
// Chromium: it wakes, loads and checks its organs, and answers as the headset does.
//
//     node tests/test_web.js SITE_DIR        (needs playwright; SITE_DIR/organs optional)
const http = require("http");
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const site = path.resolve(process.argv[2] || "web/site");
const MIME = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json",
  ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json", ".txt": "text/plain" };
const server = http.createServer((req, res) => {
  const file = path.join(site, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (!file.startsWith(site) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { "Content-Type": MIME[path.extname(file)] || "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
});

function check(ok, what) {
  console.log(`${ok ? "ok " : "NO "} ${what}`);
  if (!ok) process.exitCode = 1;
}

(async () => {
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/index.html`);
  await page.waitForFunction(() => document.querySelector("#phase")?.textContent === "PRONTA", null, { timeout: 240000 });
  const ask = async (q) => {
    const before = await page.$$eval(".msg", (e) => e.length);
    await page.fill("#input", q);
    await page.press("#input", "Enter");
    await page.waitForFunction((n) => [...document.querySelectorAll(".msg")].slice(n)
      .some((m) => !/user|note|draft/.test(m.className) && !m.querySelector(".cursor")), before, { timeout: 90000 });
    return (await page.$$eval(".msg", (e) => e.map((m) => m.innerText.replace(/\s+/g, " ")))).slice(before)
      .filter((t) => !t.startsWith("VOCÊ")).join(" || ");
  };
  const organs = fs.existsSync(path.join(site, "organs", "manifest.json"));
  check(/SOCIAL/.test(await ask("Eai")), "a greeting goes to the social expert");
  check(/\b30\b/.test(await ask("quanto é 15 por cento de 200")), "exact arithmetic");
  check(/entendi|peguei|mais/i.test(await ask("V")), "an input that says nothing gets a request to say more");
  check(/contas/.test(await ask("o que você sabe fazer?")), "it says what this body can do");
  await ask("meu nome é Ana");
  check(/Ana/.test(await ask("qual é meu nome?")), "memory");
  if (organs) check(/navegador|descobridor/i.test(await ask("quem foi Pedro Álvares Cabral")), "Wikipedia answers by the article asked about");
  const trust = JSON.parse(await page.evaluate(() => window.MLG.trust()));
  check(trust.events > 5 && trust.broken === 0, `trust chain intact (${trust.events} events)`);
  check(errors.length === 0, "no page errors" + (errors.length ? ": " + errors.join("; ") : ""));
  await browser.close();
  server.close();
  console.log(process.exitCode ? "the web MLG FAILED" : "the web MLG wakes, checks its organs and answers like the headset");
})().catch((e) => { console.error(e); process.exit(1); });
