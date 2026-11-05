"use strict";
// Acompaña la transición entre páginas con una línea de escaneo y activa los marcos de enfoque de las cámaras.
(() => {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const raiz = document.documentElement;
  raiz.classList.add("arrancando");
  setTimeout(() => raiz.classList.remove("arrancando"), 2600);   // las cámaras del panel llegan tras conectarse

  // pagereveal: la página nueva se muestra tras una navegación con transición (Chrome 123 o más)
  addEventListener("pagereveal", (e) => {
    if (!e.viewTransition) return;
    const linea = document.createElement("div");
    linea.className = "linea-escaneo";
    linea.setAttribute("aria-hidden", "true");
    (document.body || document.documentElement).append(linea);
    linea.addEventListener("animationend", () => linea.remove());
  });
})();
