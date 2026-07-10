// Cross-origin isolation shim for hosts that cannot set response headers
// (GitHub Pages). SharedArrayBuffer — and therefore wasm threads — needs
// COOP/COEP; a service worker can inject those headers onto every
// response it proxies. First visit: register the worker and reload once
// so the page loads under its scope. Already-isolated pages (or browsers
// without service workers) do nothing.
//
// Loaded as both the page script (registration side) and the service
// worker itself (fetch-proxy side); the two roles are told apart by the
// global scope.

/* eslint-env browser, serviceworker */

if (typeof window === "undefined") {
  // ---- service-worker role -------------------------------------------
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", (event) => {
    const req = event.request;
    if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
    event.respondWith(
      fetch(req).then((response) => {
        if (response.status === 0) return response;
        const headers = new Headers(response.headers);
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers,
        });
      }),
    );
  });
} else if (!window.crossOriginIsolated && "serviceWorker" in navigator) {
  // ---- page role ------------------------------------------------------
  // clients.claim() takes control of this already-loaded page without
  // re-fetching its document, so the COOP/COEP headers only apply after
  // one real reload. Reload exactly once, when control first arrives.
  let reloaded = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (!reloaded) {
      reloaded = true;
      window.location.reload();
    }
  });
  navigator.serviceWorker.register(document.currentScript.src).then(
    (registration) => {
      // Already controlled but not isolated (e.g. a stale worker):
      // refresh the worker; the controllerchange above handles the rest.
      if (navigator.serviceWorker.controller) {
        registration.update().catch(() => {});
        if (!reloaded) {
          reloaded = true;
          window.location.reload();
        }
      }
    },
    () => {
      // No isolation available; the single-threaded bundle still runs.
    },
  );
}
