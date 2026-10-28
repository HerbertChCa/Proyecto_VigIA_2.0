"use strict";
// Reto VigIA: el visitante actúa una situación y la página espera la alerta correspondiente por SSE.
const $ = (s) => document.querySelector(s);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const NOMBRE_ZONA = { restringida: "roja (restringida)", merodeo: "amarilla (merodeo)" };

let datos = { retos: [], marcador: {}, zonas: {}, emociones: false };
let camaras = [];
let elegido = null;
let fase = "elegir";            // elegir | cuenta | curso | resultado
let inicio = 0, reloj = null, cierre = null;

async function api(url, opciones = {}) {
  const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opciones });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}
const camara = () => $("#camara").value;

// ------------------------------------------------------------------ retos y marcador
function bloqueo(r) {
  if (!camara()) return "Agregue una cámara en el panel.";
  if (r.zona && !(datos.zonas[camara()] || []).includes(r.zona))
    return `Falta dibujar una zona ${NOMBRE_ZONA[r.zona]} en «Zonas» del panel.`;
  if (r.emocion && !datos.emociones) return "Active «Analizar expresiones faciales» en Ajustes del panel.";
  return null;
}
function pintarRetos() {
  const cont = $("#retos"); cont.replaceChildren();
  for (const r of datos.retos) {
    const m = datos.marcador[r.id] || {}, falta = bloqueo(r);
    const b = el("button", "reto-tarjeta" + (elegido?.id === r.id ? " elegido" : "") + (falta ? " bloqueado" : ""));
    b.append(el("b", null, r.titulo));
    b.append(el("small", null, m.intentos
      ? `Detectado en ${m.detectados} de ${m.intentos} intentos (${m.tasa} %)` + (m.mediana_s != null ? ` · ${m.mediana_s} s típico` : "")
      : "Nadie lo ha intentado todavía"));
    if (falta) b.append(el("small", "falta", falta));
    b.onclick = () => { if (fase === "elegir" || fase === "resultado") elegir(r); };
    cont.append(b);
  }
  const tot = Object.values(datos.marcador).reduce((a, m) => [a[0] + m.detectados, a[1] + m.intentos], [0, 0]);
  const g = $("#global"); g.replaceChildren();
  g.append(el("b", null, tot[1] ? `${Math.round((100 * tot[0]) / tot[1])} %` : "—"));
  g.append(el("span", null, tot[1]
    ? `de los ${tot[1]} intentos de hoy fueron detectados. Es una medida real de la sensibilidad de VigIA con personas, no una cifra de laboratorio.`
    : "Aquí se verá cuántos intentos detecta VigIA con personas reales."));
}
function elegir(r) {
  if (elegido?.id !== r.id) $("#btn-empezar").textContent = "¡Empezar!";
  elegido = r; fase = "elegir"; clearTimeout(cierre); ocultarCapa();
  $("#consigna-titulo").textContent = r.titulo;
  const falta = bloqueo(r);
  $("#consigna-texto").textContent = falta ? falta : `${r.instruccion} Tienes ${r.limite} s.`;
  $("#btn-empezar").disabled = !!falta;
  pintarRetos();
}

// ------------------------------------------------------------------ capa sobre el video
function mostrarCapa(grande, texto, clase = "") {
  const c = $("#capa"); c.hidden = false; c.className = "capa " + clase;
  $("#capa-grande").textContent = grande;
  const t = $("#capa-texto"); t.replaceChildren();
  if (typeof texto === "string") t.textContent = texto; else if (texto) t.append(texto);
}
function ocultarCapa() { $("#capa").hidden = true; $("#barra-tiempo").hidden = true; }

// ------------------------------------------------------------------ desarrollo de un intento
function empezar() {
  if (!elegido || bloqueo(elegido)) return;
  fase = "cuenta"; clearTimeout(cierre);
  $("#btn-empezar").hidden = true; $("#btn-cancelar").hidden = false;
  let n = 3;
  mostrarCapa(n, "¡Prepárate!");
  const paso = setInterval(() => {
    n -= 1;
    if (fase !== "cuenta") return clearInterval(paso);
    if (n > 0) return mostrarCapa(n, "¡Prepárate!");
    clearInterval(paso);
    fase = "curso"; inicio = performance.now(); ocultarCapa();
    $("#barra-tiempo").hidden = false;
    reloj = setInterval(tic, 200);
  }, 1000);
}
function tic() {
  const s = (performance.now() - inicio) / 1000, quedan = Math.max(0, elegido.limite - s);
  $("#barra-relleno").style.width = `${(100 * quedan) / elegido.limite}%`;
  $("#consigna-texto").textContent = `${elegido.instruccion} Quedan ${Math.ceil(quedan)} s.`;
  if (quedan <= 0) terminar(false);
}
async function terminar(detectado, alerta = null) {
  if (fase !== "curso") return;
  fase = "resultado"; clearInterval(reloj);
  const s = (performance.now() - inicio) / 1000;
  $("#barra-tiempo").hidden = true;
  $("#btn-empezar").hidden = false; $("#btn-cancelar").hidden = true; $("#btn-empezar").textContent = "Otra vez";
  if (detectado) {
    const caja = el("div");
    caja.append(el("div", null, alerta ? alerta.mensaje : "VigIA reconoció tu expresión."));
    if (alerta?.explicacion) caja.append(el("div", "explicacion", `Por qué: ${alerta.explicacion}`));
    mostrarCapa(`¡Detectado en ${s.toFixed(1)} s!`, caja, "detectado");
  } else {
    mostrarCapa("No te detectó", `Pasaron ${elegido.limite} s sin alerta. Las reglas tienen límites: con poca luz, de espaldas o tapado por otra persona, el esqueleto se pierde.`, "escapo");
  }
  $("#consigna-texto").textContent = elegido.instruccion;
  try { datos.marcador = await api("/api/reto/resultado", { method: "POST", body: JSON.stringify({ reto: elegido.id, detectado, segundos: detectado ? s : null }) }); }
  catch (e) { console.error(e); }
  pintarRetos();
  cierre = setTimeout(() => { if (fase === "resultado") { fase = "elegir"; ocultarCapa(); } }, 15000);
}
function cancelar() {
  fase = "elegir"; clearInterval(reloj); ocultarCapa();
  $("#btn-empezar").hidden = false; $("#btn-cancelar").hidden = true; $("#btn-empezar").textContent = "¡Empezar!";
  if (elegido) elegir(elegido);
}

// ------------------------------------------------------------------ eventos en vivo
function conectar() {
  const es = new EventSource("/api/eventos");
  es.onmessage = (m) => {
    const { evento, datos: d } = JSON.parse(m.data);
    if (evento === "estado") {
      actualizarCamaras(d.camaras);
      $("#btn-esqueleto").setAttribute("aria-pressed", String(!!d.solo_esqueleto));
      if (fase === "curso" && elegido?.emocion) {
        const c = d.camaras.find((x) => x.id === camara());
        if (c && (c.emociones || {})[elegido.emocion] > 0) terminar(true);
      }
    } else if (evento === "alerta" && fase === "curso" && elegido?.tipos && d.camara === camara() && elegido.tipos.includes(d.tipo)) {
      terminar(true, d);
    }
  };
  es.onerror = () => { es.close(); setTimeout(conectar, 2000); };
}
function actualizarCamaras(lista) {
  const ids = lista.map((c) => c.id).join();
  if (ids === camaras.map((c) => c.id).join()) return;
  camaras = lista;
  const sel = $("#camara"), previa = sel.value;
  sel.replaceChildren(...lista.map((c) => { const o = el("option", null, c.nombre); o.value = c.id; return o; }));
  if (lista.some((c) => c.id === previa)) sel.value = previa;
  cambiarCamara();
}
function cambiarCamara() {
  $("#video").src = camara() ? `/video/${camara()}` : "";
  if (elegido) elegir(elegido); else pintarRetos();
}

// ------------------------------------------------------------------ inicio
$("#btn-empezar").onclick = empezar;
$("#btn-cancelar").onclick = cancelar;
$("#camara").onchange = cambiarCamara;
$("#btn-esqueleto").onclick = async () => {
  const activo = $("#btn-esqueleto").getAttribute("aria-pressed") !== "true";
  $("#btn-esqueleto").setAttribute("aria-pressed", String(activo));
  await api("/api/ajustes", { method: "PUT", body: JSON.stringify({ privacidad: { solo_esqueleto: activo } }) });
};
$("#qr").src = "/api/encuesta/qr.png";
(async () => {
  datos = await api("/api/reto");
  pintarRetos();
  conectar();
  setInterval(async () => {          // zonas y ajustes pueden cambiar desde el panel
    const d = await api("/api/reto").catch(() => null);
    if (!d || fase === "curso" || fase === "cuenta") return;
    datos = d;
    if (fase === "elegir" && elegido) elegir(datos.retos.find((r) => r.id === elegido.id)); else pintarRetos();
  }, 5000);
})();
