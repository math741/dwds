// Keeps the MLG working offline once it has been opened: organ parts are kept by
// their hash (they never change under the same name and ?v=), everything else is
// asked of the network first and kept for when there is none.
const CACHE = "mlg-site";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  const organ = url.pathname.includes("/organs/") && url.searchParams.has("v");
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    if (organ) {
      const hit = await cache.match(e.request);
      if (hit) return hit;
      const r = await fetch(e.request);
      if (r.ok) cache.put(e.request, r.clone()).catch(() => {});
      return r;
    }
    try {
      const r = await fetch(e.request);
      if (r.ok) cache.put(e.request, r.clone()).catch(() => {});
      return r;
    } catch (err) {
      const hit = await cache.match(e.request);
      if (hit) return hit;
      throw err;
    }
  })());
});
