"use strict";
// Encuesta de aceptación (celular). Las preguntas vienen del servidor (vigia/encuesta.py).
const $ = (s) => document.querySelector(s);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
let PREGUNTAS = [], LIKERT = [];

function terminar(titulo, texto, enlace = true) {
  $("#paso-edad").hidden = true; $("#form").hidden = true;
  const m = $("#mensaje"); m.hidden = false; m.replaceChildren(el("b", null, titulo), el("p", null, texto));
  if (enlace) { const a = el("a", null, "Ver los resultados"); a.href = "/encuesta/resultados"; m.append(a); }
  scrollTo(0, 0);
}

function construir() {
  const f = $("#form"); f.replaceChildren();
  for (const p of PREGUNTAS) {
    const caja = el("fieldset", "pregunta");
    const leg = el("legend"); leg.append(el("span", "bloque", p.bloque), document.createTextNode(p.texto));
    caja.append(leg);
    const ops = el("div", "opciones" + (p.tipo === "likert" ? " escala" : ""));
    if (p.tipo === "likert") {
      LIKERT.forEach((txt, i) => {
        const l = el("label"), r = el("input"); r.type = "radio"; r.name = p.id; r.value = i + 1;
        l.append(r, el("b", null, i + 1), el("span", null, txt)); ops.append(l);
      });
    } else {
      for (const o of p.opciones) {
        const l = el("label"), r = el("input"); r.type = p.tipo === "varias" ? "checkbox" : "radio"; r.name = p.id; r.value = o;
        l.append(r, el("span", null, o)); ops.append(l);
      }
    }
    caja.append(ops); f.append(caja);
  }
  const b = el("button", "btn primario grande enviar", "Enviar respuestas"); b.type = "submit"; f.append(b);
  f.hidden = false;
}

document.querySelectorAll('input[name="mayor"]').forEach((r) => (r.onchange = () => {
  if (r.value === "no") return terminar("Gracias por tu interés", "Esta encuesta es solo para mayores de 18 años. No se guardó ninguna respuesta.", false);
  $("#paso-edad").hidden = true; construir();
}));

$("#form").onsubmit = async (e) => {
  e.preventDefault();
  const resp = { mayor_de_edad: true };
  for (const p of PREGUNTAS) {
    const marcadas = [...document.querySelectorAll(`[name="${p.id}"]:checked`)].map((i) => i.value);
    if (!marcadas.length) continue;
    resp[p.id] = p.tipo === "likert" ? Number(marcadas[0]) : p.tipo === "varias" ? marcadas : marcadas[0];
  }
  if (Object.keys(resp).length < 2) return alert("Responde al menos una pregunta.");
  const b = e.target.querySelector("button"); b.disabled = true;
  try {
    const r = await fetch("/api/encuesta", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(resp) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || "No se pudo enviar");
    try { localStorage.setItem("vigia_encuesta", "1"); } catch {}
    terminar("¡Gracias!", "Tus respuestas se sumaron a los resultados que se ven en la pantalla del stand.");
  } catch (err) { alert(err.message); b.disabled = false; }
};

(async () => {
  const d = await (await fetch("/api/encuesta/preguntas")).json();
  PREGUNTAS = d.preguntas; LIKERT = d.likert;
  let ya = false; try { ya = localStorage.getItem("vigia_encuesta") === "1"; } catch {}
  if (ya) terminar("Ya respondiste desde este celular", "¡Gracias! Puedes ver cómo van los resultados.");
})();
