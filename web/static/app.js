"use strict";
// Panel de VigIA: video en vivo (MJPEG), alertas en tiempo real (Server-Sent Events), zonas y ajustes.

const TIPOS = {
  violencia: "Violencia", caida: "Caída", en_suelo: "Persona en el suelo", merodeo: "Merodeo",
  intrusion: "Intrusión", abandono: "Objeto abandonado", hurto: "Posible hurto", emocion: "Emoción",
};
const $ = (s) => document.querySelector(s);
const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };

let estado = { camaras: [], resumen: {} };
let filtro = "";
const alertasCargadas = new Map();
const camaraEnAlerta = new Map();

// ------------------------------------------------------------------ utilidades
const hora = (iso) => iso.slice(11, 19);
const fechaCorta = (iso) => `${iso.slice(8, 10)}/${iso.slice(5, 7)} ${hora(iso)}`;
async function api(url, opciones = {}) {
  const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...opciones });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}
let audio;
function pitido() {
  if (!$("#sonido").checked) return;
  audio = audio || new AudioContext();
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = 880; g.gain.value = 0.08;
  o.connect(g).connect(audio.destination); o.start(); o.stop(audio.currentTime + 0.25);
}

// ------------------------------------------------------------------ cámaras
function pintarCamaras() {
  const cont = $("#camaras");
  const ids = estado.camaras.map((c) => c.id);
  [...cont.children].forEach((n) => { if (!ids.includes(n.dataset.id)) n.remove(); });
  for (const c of estado.camaras) {
    let tarjeta = cont.querySelector(`[data-id="${c.id}"]`);
    if (!tarjeta) {
      tarjeta = el("article", "camara"); tarjeta.dataset.id = c.id;
      const cab = el("div", "camara-cab");
      cab.append(el("h3", null, c.nombre), el("span", "meta"));
      const quitar = el("button", "quitar", "✕"); quitar.title = "Quitar cámara";
      quitar.onclick = async () => { if (confirm(`¿Quitar la cámara «${c.nombre}»?`)) { await api(`/api/camaras/${c.id}`, { method: "DELETE" }); } };
      cab.append(quitar);
      const video = el("div", "video");
      const img = el("img"); img.alt = `Video de ${c.nombre}`; img.src = `/video/${c.id}?t=${Date.now()}`;
      img.onerror = () => setTimeout(() => { img.src = `/video/${c.id}?t=${Date.now()}`; }, 2000);
      video.append(img, el("div", "estado"));
      tarjeta.append(cab, video, el("div", "emociones"));
      cont.append(tarjeta);
    }
    tarjeta.querySelector(".meta").textContent = `${c.fps.toFixed(1)} fps · ${c.personas} personas`;
    const aviso = tarjeta.querySelector(".estado");
    aviso.textContent = c.error ? `Error: ${c.error}` : c.conectada ? "" : "Conectando con la cámara…";
    aviso.hidden = !aviso.textContent;
    const emo = tarjeta.querySelector(".emociones"); emo.replaceChildren();
    const lista = Object.entries(c.emociones || {});
    if (lista.length) lista.forEach(([k, n]) => emo.append(el("span", "chip", `${k}: ${n}`)));
    else emo.append(el("span", "chip", "sin expresiones detectadas"));
    tarjeta.classList.toggle("en-alerta", Date.now() - (camaraEnAlerta.get(c.id) || 0) < 4000);
  }
}

// ------------------------------------------------------------------ resumen e indicadores
function pintarResumen() {
  const cont = $("#resumen"); cont.replaceChildren();
  let pendientes = 0;
  for (const [tipo, nombre] of Object.entries(TIPOS)) {
    const r = estado.resumen[tipo] || { total: 0, pendientes: 0 };
    pendientes += r.pendientes;
    const t = el("button", "tarjeta" + (filtro === tipo ? " activa" : ""));
    const b = el("b", null, r.total);
    if (r.pendientes) b.append(el("em", null, `${r.pendientes} pend.`));
    t.append(b, el("span", null, nombre));
    t.onclick = () => { filtro = filtro === tipo ? "" : tipo; $("#filtro-tipo").value = filtro; cargarAlertas(); pintarResumen(); };
    cont.append(t);
  }
  $("#ind-pendientes").textContent = pendientes;
  $("#ind-camaras").textContent = estado.camaras.length;
  $("#ind-personas").textContent = estado.camaras.reduce((s, c) => s + c.personas, 0);
  $("#aviso-demo").hidden = !estado.demo;
}

// ------------------------------------------------------------------ alertas
function tarjetaAlerta(a, nueva) {
  const li = el("li", "alerta" + (a.atendida ? " atendida" : "") + (nueva ? " nueva" : ""));
  li.dataset.id = a.id;
  const img = el("img"); img.src = `/capturas/${a.captura}`; img.alt = "Captura"; img.loading = "lazy";
  img.onclick = () => abrirCaptura(a);
  const cuerpo = el("div");
  cuerpo.append(el("div", "tipo", `${a.tipo_nombre} · ${a.camara}`), el("div", "msj", a.mensaje), el("div", "det", a.explicacion));
  const pie = el("div", "pie-alerta");
  pie.append(el("span", "cuando", fechaCorta(a.fecha)));
  pie.append(accionesAlerta(a));
  cuerpo.append(pie);
  li.append(el("div", `sev ${a.severidad}`), img, cuerpo);
  return li;
}
// El veredicto del operador mide la precisión real de cada instalación (artículo, Tabla 11).
function accionesAlerta(a) {
  const caja = el("span", "acciones-alerta");
  if (a.veredicto) {
    caja.append(el("span", `veredicto ${a.veredicto}`, a.veredicto === "confirmada" ? "Confirmada" : "Falsa alarma"));
  } else if (a.atendida) {
    caja.append(el("span", "veredicto", "Atendida"));
  } else {
    const si = el("button", "btn chico confirmar", "Confirmar"), no = el("button", "btn chico", "Falsa alarma");
    si.title = "La situación ocurrió de verdad"; no.title = "La alerta fue un error del sistema";
    const enviar = (valor) => api(`/api/alertas/${a.id}/veredicto`, { method: "POST", body: JSON.stringify({ valor }) });
    si.onclick = () => enviar("confirmada"); no.onclick = () => enviar("falsa");
    caja.append(si, no);
  }
  return caja;
}
function pintarOperador() {
  const o = (estado.operador || {}).total || {};
  const cont = $("#operador"); cont.replaceChildren();
  const revisadas = (o.confirmadas || 0) + (o.falsas || 0);
  const txt = revisadas
    ? `Revisión del operador: ${o.confirmadas} ${o.confirmadas === 1 ? "confirmada" : "confirmadas"} y ${o.falsas} ${o.falsas === 1 ? "falsa alarma" : "falsas alarmas"}. Precisión observada en esta instalación: ${(o.precision * 100).toFixed(0)} % (n = ${revisadas}).`
    : "Revisión del operador: marque cada alerta como «Confirmar» o «Falsa alarma» para medir la precisión real en esta instalación.";
  const a = el("a", "btn chico", "Exportar CSV"); a.href = "/api/alertas.csv";
  cont.append(el("span", null, txt), a);
}
function pintarAlertas() {
  const lista = $("#alertas"); lista.replaceChildren();
  const vis = [...alertasCargadas.values()].filter((a) => !filtro || a.tipo === filtro).sort((x, y) => y.id - x.id);
  vis.forEach((a) => lista.append(tarjetaAlerta(a, false)));
  $("#sin-alertas").hidden = vis.length > 0;
}
async function cargarAlertas() {
  const datos = await api(`/api/alertas?limite=150${filtro ? `&tipo=${filtro}` : ""}`);
  if (!filtro) alertasCargadas.clear();
  datos.forEach((a) => alertasCargadas.set(a.id, a));
  pintarAlertas();
}
function llegaAlerta(a) {
  alertasCargadas.set(a.id, a);
  camaraEnAlerta.set(a.camara, Date.now());
  if (a.severidad === "alta") pitido();
  if (filtro && a.tipo !== filtro) return;
  $("#alertas").prepend(tarjetaAlerta(a, true));
  $("#sin-alertas").hidden = true;
}
function abrirCaptura(a) {
  $("#cap-titulo").textContent = `${a.tipo_nombre} · ${a.camara} · ${fechaCorta(a.fecha)}`;
  $("#cap-img").src = `/capturas/${a.captura}`;
  $("#cap-texto").textContent = `${a.mensaje}. ${a.explicacion}`;
  $("#dlg-captura").showModal();
}

// ------------------------------------------------------------------ tiempo real
function conectar() {
  const ind = $("#conexion");
  const fuente = new EventSource("/api/eventos");
  fuente.onopen = () => { ind.className = "conexion ok"; ind.querySelector("span").textContent = "en línea"; };
  fuente.onerror = () => { ind.className = "conexion caida"; ind.querySelector("span").textContent = "reconectando"; };
  fuente.onmessage = (m) => {
    const { evento, datos } = JSON.parse(m.data);
    if (evento === "estado") { estado = datos; pintarCamaras(); pintarResumen(); pintarOperador(); }
    else if (evento === "alerta") llegaAlerta(datos);
    else if (evento === "atendida") {
      const a = alertasCargadas.get(datos.id);
      if (a) { a.atendida = 1; a.veredicto = datos.veredicto; }
      const li = document.querySelector(`.alerta[data-id="${datos.id}"]`);
      if (li && a) { li.classList.add("atendida"); li.querySelector(".acciones-alerta").replaceWith(accionesAlerta(a)); }
    }
  };
}

// ------------------------------------------------------------------ nueva cámara
$("#btn-camara").onclick = () => $("#dlg-camara").showModal();
$("#form-camara").onsubmit = async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  try {
    await api("/api/camaras", { method: "POST", body: JSON.stringify({ nombre: f.get("nombre"), fuente: f.get("fuente") }) });
    e.target.reset(); $("#dlg-camara").close();
  } catch (err) { alert(err.message); }
};

// ------------------------------------------------------------------ ajustes
const leer = (o, ruta) => ruta.split(".").reduce((x, k) => x?.[k], o);
function escribir(o, ruta, v) { const k = ruta.split("."); const u = k.pop(); k.reduce((x, c) => (x[c] ??= {}), o)[u] = v; }
$("#btn-ajustes").onclick = async () => {
  const aj = await api("/api/ajustes");
  document.querySelectorAll("#form-ajustes [data-ruta]").forEach((i) => {
    const v = leer(aj, i.dataset.ruta);
    if (i.type === "checkbox") i.checked = !!v; else i.value = v;
  });
  $("#dias-retencion").textContent = aj.privacidad.dias_retencion_capturas;
  $("#dlg-ajustes").showModal();
};
$("#form-ajustes").onsubmit = async (e) => {
  e.preventDefault();
  const cambios = {};
  document.querySelectorAll("#form-ajustes [data-ruta]").forEach((i) =>
    escribir(cambios, i.dataset.ruta, i.type === "checkbox" ? i.checked : Number(i.value)));
  await api("/api/ajustes", { method: "PUT", body: JSON.stringify(cambios) });
  $("#dlg-ajustes").close();
};

// ------------------------------------------------------------------ editor de zonas
const lienzo = $("#zona-lienzo"), ctx = lienzo.getContext("2d");
let zonas = [], puntos = [], camZona = null;
const COLORES = { restringida: "#e5534b", merodeo: "#e3a33a" };

function ajustarLienzo() {
  const r = lienzo.getBoundingClientRect(), d = devicePixelRatio || 1;
  lienzo.width = r.width * d; lienzo.height = r.height * d;
  dibujarZonas();
}
function dibujarZonas() {
  const W = lienzo.width, H = lienzo.height;
  ctx.clearRect(0, 0, W, H);
  ctx.lineWidth = 2 * (devicePixelRatio || 1);
  ctx.font = `${13 * (devicePixelRatio || 1)}px Segoe UI, sans-serif`;
  for (const z of zonas) {
    const col = COLORES[z.tipo];
    ctx.beginPath(); z.puntos.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H))); ctx.closePath();
    ctx.fillStyle = col + "33"; ctx.fill(); ctx.strokeStyle = col; ctx.stroke();
    ctx.fillStyle = col; ctx.fillText(z.nombre, z.puntos[0][0] * W + 6, z.puntos[0][1] * H + 16);
  }
  if (puntos.length) {
    ctx.strokeStyle = "#ffffff"; ctx.beginPath();
    puntos.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H))); ctx.stroke();
    ctx.fillStyle = "#ffffff"; puntos.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x * W, y * H, 4, 0, 7); ctx.fill(); });
  }
}
function pintarListaZonas() {
  const ul = $("#zona-lista"); ul.replaceChildren();
  zonas.forEach((z, i) => {
    const li = el("li", null, `${z.nombre} · ${z.tipo}`);
    const b = el("button", null, "✕"); b.title = "Eliminar zona";
    b.onclick = () => { zonas.splice(i, 1); pintarListaZonas(); dibujarZonas(); };
    li.append(b); ul.append(li);
  });
}
async function cargarZonas() {
  camZona = $("#zona-camara").value;
  zonas = await api(`/api/camaras/${camZona}/zonas`); puntos = [];
  $("#zona-fondo").src = `/api/captura/${camZona}?crudo=1&t=${Date.now()}`;
  pintarListaZonas(); requestAnimationFrame(ajustarLienzo);
}
$("#btn-zonas").onclick = async () => {
  const sel = $("#zona-camara"); sel.replaceChildren();
  estado.camaras.forEach((c) => { const o = el("option", null, c.nombre); o.value = c.id; sel.append(o); });
  if (!estado.camaras.length) return alert("Primero agregue una cámara.");
  $("#dlg-zonas").showModal(); await cargarZonas();
};
$("#zona-camara").onchange = cargarZonas;
$("#zona-fondo").onload = ajustarLienzo;
lienzo.onclick = (e) => {
  const r = lienzo.getBoundingClientRect();
  puntos.push([(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]); dibujarZonas();
};
lienzo.ondblclick = () => {
  // el doble clic también disparó dos clics: se quita el punto repetido
  while (puntos.length > 1) {
    const [a, b] = [puntos.at(-1), puntos.at(-2)];
    if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 0.01) puntos.pop(); else break;
  }
  if (puntos.length < 3) return;
  zonas.push({ nombre: $("#zona-nombre").value || "Zona", tipo: $("#zona-tipo").value, puntos });
  puntos = []; $("#zona-nombre").value = `Zona ${zonas.length + 1}`;
  pintarListaZonas(); dibujarZonas();
};
$("#zona-deshacer").onclick = () => { puntos.pop(); dibujarZonas(); };
$("#zona-guardar").onclick = async () => {
  zonas = await api(`/api/camaras/${camZona}/zonas`, { method: "PUT", body: JSON.stringify(zonas) });
  $("#dlg-zonas").close();
};
addEventListener("resize", () => { if ($("#dlg-zonas").open) ajustarLienzo(); });

// ------------------------------------------------------------------ inicio
const sel = $("#filtro-tipo");
Object.entries(TIPOS).forEach(([k, v]) => { const o = el("option", null, v); o.value = k; sel.append(o); });
sel.onchange = () => { filtro = sel.value; cargarAlertas(); pintarResumen(); };
cargarAlertas().catch(console.error);
conectar();
setInterval(pintarCamaras, 1000);
