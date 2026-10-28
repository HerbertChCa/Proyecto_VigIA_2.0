"use strict";
// Resultados en vivo de la encuesta: se actualizan cada 4 s.
const $ = (s) => document.querySelector(s);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const COLORES_LIKERT = ["#e5534b", "#e8916b", "#6b7380", "#7fc9bd", "#3fb6a8"];
const ETIQUETAS = ["Muy en desacuerdo", "En desacuerdo", "Neutral", "De acuerdo", "Muy de acuerdo"];

function filaBarra(texto, pct) {
  const f = el("div", "fila-barra"), pista = el("div", "pista"), i = el("i");
  i.style.width = `${pct}%`; pista.append(i);
  f.append(el("span", null, texto), pista, el("span", null, `${Math.round(pct)} %`));
  return f;
}

function tarjeta(p) {
  const t = el("div", "res-tarjeta");
  t.append(el("span", "bloque", p.bloque), el("h3", null, p.texto));
  if (!p.n) { t.append(el("p", "nota", "Sin respuestas todavía.")); return t; }
  if (p.tipo === "likert") {
    const ley = el("div", "leyenda");
    ETIQUETAS.forEach((e, k) => { const s = el("span"); const c = el("i"); c.style.background = COLORES_LIKERT[k]; s.append(c, e); ley.append(s); });
    const pila = el("div", "likert-pila");
    p.conteo.forEach((c, k) => { const i = el("i"); i.style.width = `${(100 * c) / p.n}%`; i.style.background = COLORES_LIKERT[k]; i.title = `${ETIQUETAS[k]}: ${c}`; pila.append(i); });
    const ac = el("div", "acuerdo");
    const a = el("div"); a.append(el("b", null, `${p.acuerdo} %`), document.createTextNode(" de acuerdo"));
    ac.append(a);
    if (p.ref) { const r = el("div"); r.append(el("b", null, `${p.ref.valor} %`), document.createTextNode(` ${p.ref.fuente}`)); ac.append(r); }
    t.append(ley, pila, ac);
  } else {
    for (const [op, pct] of Object.entries(p.porcentaje)) t.append(filaBarra(op, pct));
  }
  t.append(el("p", "nota", `${p.n} respuestas`));
  return t;
}

async function actualizar() {
  const r = await fetch("/api/encuesta/resultados");
  if (!r.ok) return;
  const d = await r.json();
  $("#n").textContent = d.n;
  $("#url").textContent = d.url;
  $("#aviso-red").hidden = d.feria;
  $("#lista").replaceChildren(...d.preguntas.map(tarjeta));
}

// desde un celular solo se muestran los resultados (sin exportar ni enlaces del stand)
if (!["127.0.0.1", "localhost", "[::1]"].includes(location.hostname)) {
  $("#acciones-local").remove(); $("#qr-caja").remove();
}
actualizar(); setInterval(actualizar, 4000);
