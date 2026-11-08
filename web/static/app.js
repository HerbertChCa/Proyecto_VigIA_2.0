"use strict";
// Panel de VigIA: video en vivo (MJPEG), alertas en tiempo real (Server-Sent Events), zonas y ajustes.

const TIPOS = {
  violencia: "Violencia", caida: "Caída", en_suelo: "Persona en el suelo", merodeo: "Merodeo",
  intrusion: "Intrusión", abandono: "Objeto abandonado", hurto: "Posible hurto", robo_vehiculo: "Robo de vehículo",
  robo_huida: "Robo con huida", vandalismo: "Vandalismo", arma: "Posible arma", aglomeracion: "Aglomeración",
  estampida: "Huida masiva", emocion: "Emoción",
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
      cab.append(el("h3", null, c.nombre), el("span", "meta"), el("span", "precision"));
      const quitar = el("button", "quitar", "✕"); quitar.title = "Quitar cámara";
      quitar.onclick = async () => {
        if (!confirm(`¿Quitar la cámara «${c.nombre}»?`)) return;
        quitar.disabled = true;
        try {
          await api(`/api/camaras/${c.id}`, { method: "DELETE" });
        } catch (e) {
          if (!/ya no existe/.test(e.message)) { quitar.disabled = false; alert(`No se pudo quitar la cámara: ${e.message}`); return; }
        }
        // sin esperar al siguiente estado del servidor: si la conexión en vivo está caída, la tarjeta igual desaparece
        estado.camaras = estado.camaras.filter((x) => x.id !== c.id);
        tarjeta.remove();
      };
      // Pausa: solo en videos subidos (y en el modo demo); una cámara en vivo no se detiene
      const pausa = el("button", "pausa", "⏸"); pausa.hidden = true;
      pausa.onclick = async () => {
        const ahora = estado.camaras.find((x) => x.id === c.id) || {};
        const quiere = ahora.terminado ? false : !ahora.pausado;     // terminado: ▶ lo vuelve a empezar
        pausa.disabled = true;
        try {
          await api(`/api/camaras/${c.id}/pausa`, { method: "POST", body: JSON.stringify({ pausado: quiere }) });
          const actual = estado.camaras.find((x) => x.id === c.id);
          if (actual) { actual.pausado = quiere; if (!quiere) actual.terminado = false; }   // sin esperar al servidor
          pintarCamaras();
        } catch (e) {
          alert(`No se pudo ${quiere ? "pausar" : "reanudar"} el video: ${e.message}`);
        } finally {
          pausa.disabled = false;
        }
      };
      // Video subido: «Detallado» analiza cada segundo completo (cámara lenta si la PC no alcanza);
      // «Tiempo real» lo pasa a su velocidad y se salta lo que no alcanza a ver, igual que una cámara en vivo
      const modo = el("button", "modo"); modo.hidden = true;
      modo.onclick = async () => {
        const ahora = estado.camaras.find((x) => x.id === c.id) || {};
        const quiere = !ahora.detallado;
        modo.disabled = true;
        try {
          await api(`/api/camaras/${c.id}/modo`, { method: "POST", body: JSON.stringify({ detallado: quiere }) });
          if (ahora.id) ahora.detallado = quiere;
          pintarCamaras();
        } catch (e) {
          alert(`No se pudo cambiar el modo del video: ${e.message}`);
        } finally {
          modo.disabled = false;
        }
      };
      cab.append(modo, pausa, quitar);
      const video = el("div", "video");
      const img = el("img"); img.alt = `Video de ${c.nombre}`;
      // Cada cuadro se pide por separado en vez de mantener un video MJPEG abierto: el navegador solo admite
      // 6 conexiones abiertas a la vez con el mismo servidor, y con 5 cámaras y las alertas en vivo se llenaban.
      // Entonces cualquier otro pedido (quitar una cámara, las imágenes de las alertas) quedaba esperando sin fin.
      const cargar = () => {
        if (!img.isConnected) return;                          // la cámara se quitó
        if (document.hidden) { setTimeout(cargar, 1000); return; }
        img.src = `/api/captura/${c.id}?t=${Date.now()}`;
      };
      img.onload = () => setTimeout(cargar, 100);
      img.onerror = () => setTimeout(cargar, 1000);             // aún sin imagen o cámara caída
      setTimeout(cargar, 0);
      video.append(img, el("div", "estado"));
      tarjeta.append(cab, video, el("div", "emociones"));
      cont.append(tarjeta);
    }
    // un video subido se analiza completo aunque la PC sea lenta: entonces se ve en cámara lenta
    const lento = c.velocidad != null && c.velocidad < 0.95 ? ` · cámara lenta ×${c.velocidad.toFixed(2)}` : "";
    tarjeta.querySelector(".meta").textContent = c.terminado ? `video terminado · ${c.personas} personas`
      : c.pausado ? `en pausa · ${c.personas} personas`
      : `${c.fps.toFixed(1)} fps${lento} · ${c.personas} personas`;
    const modo = tarjeta.querySelector(".modo");
    modo.hidden = c.detallado == null;
    modo.textContent = c.detallado ? "Detallado" : "Tiempo real";
    modo.title = c.detallado
      ? "Analiza cada segundo del video completo (va en cámara lenta si la PC no alcanza). Clic: tiempo real"
      : "El video va a su velocidad y se salta lo que la PC no alcanza a analizar. Clic: detallado";
    modo.classList.toggle("activa", !!c.detallado);
    // precisión esperada: cuántos cuadros se analizan por cada segundo de lo que pasa frente a la cámara
    const prec = tarjeta.querySelector(".precision");
    prec.hidden = !c.precision;
    if (c.precision) {
      prec.textContent = `precisión ${c.precision}`;
      prec.className = `precision ${c.precision}`;
      const etapas = Object.entries(c.etapas || {}).map(([k, v]) => `${k}: ${v} ms`).join(", ");
      const acel = c.aceleracion ? ` · modelos: personas en ${c.aceleracion.personas}, objetos en ${c.aceleracion.objetos}` : "";
      prec.title = {
        alta: "7 o más cuadros por segundo de acción: se ven los golpes y los tirones.",
        media: "4 a 7 cuadros por segundo: se pueden perder los golpes más rápidos.",
        baja: "Menos de 4 cuadros por segundo: los tirones y golpes pueden caer entre dos cuadros. " +
          "Use el perfil rápido (python -m vigia --rapido), acelere los modelos o deje una sola cámara.",
      }[c.precision] + `\nCada cuadro: ${etapas}${acel}`;
    }
    const boton = tarjeta.querySelector(".pausa");
    boton.hidden = !c.pausable;
    boton.textContent = c.terminado ? "↻" : c.pausado ? "▶" : "⏸";
    boton.title = c.terminado ? "Ver el video otra vez" : c.pausado ? "Reanudar el video" : "Pausar el video";
    boton.classList.toggle("activa", !!(c.pausado || c.terminado));
    const aviso = tarjeta.querySelector(".estado");
    aviso.textContent = c.error ? `Error: ${c.error}` : !c.conectada ? "Conectando con la cámara…"
      : c.terminado ? "Fin del video · ↻ para verlo otra vez" : c.pausado ? "⏸ En pausa" : "";
    aviso.classList.toggle("pausado", !c.error && c.conectada && !!(c.pausado || c.terminado));
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
  const tipo = el("div", "tipo", `${a.tipo_nombre} · ${a.camara}`);
  if (a.repeticiones > 1) tipo.append(el("span", "veces", `×${a.repeticiones}`));
  cuerpo.append(tipo, el("div", "msj", a.mensaje), el("div", "det", a.explicacion));
  const pie = el("div", "pie-alerta");
  pie.append(el("span", "cuando", a.repeticiones > 1 && a.ultima ? `${fechaCorta(a.fecha)} – ${hora(a.ultima)}` : fechaCorta(a.fecha)));
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
  if (alertasCargadas.size > 500) alertasCargadas.delete(alertasCargadas.keys().next().value);
  camaraEnAlerta.set(a.camara, Date.now());
  if (a.severidad === "alta") pitido();
  if (filtro && a.tipo !== filtro) return;
  const lista = $("#alertas");
  lista.prepend(tarjetaAlerta(a, true));
  while (lista.children.length > 150) lista.lastElementChild.remove();   // con horas de uso, el panel no se vuelve pesado
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
    if (evento === "estado") {
      estado = datos; pintarCamaras(); pintarResumen(); pintarOperador();
      $("#btn-esqueleto").setAttribute("aria-pressed", String(!!datos.solo_esqueleto));
    }
    else if (evento === "alerta") llegaAlerta(datos);
    else if (evento === "repetida") {
      // la misma situación volvió a ocurrir: se suma a la alerta pendiente, que sube al inicio de la lista y se
      // resalta como una nueva (en la prueba de clase un segundo intento pasaba desapercibido como «×2»)
      const a = alertasCargadas.get(datos.id);
      if (a) {
        a.repeticiones = datos.repeticiones; a.ultima = datos.ultima; a.mensaje = datos.mensaje; a.explicacion = datos.explicacion;
        const li = document.querySelector(`.alerta[data-id="${datos.id}"]`);
        if (li) { li.remove(); $("#alertas").prepend(tarjetaAlerta(a, true)); }
        if (datos.severidad === "alta") pitido();
      }
      camaraEnAlerta.set(datos.camara, Date.now());
    }
    else if (evento === "limpiar") {
      alertasCargadas.clear(); camaraEnAlerta.clear(); pintarAlertas();
    }
    else if (evento === "atendida") {
      const a = alertasCargadas.get(datos.id);
      if (a) { a.atendida = 1; a.veredicto = datos.veredicto; }
      const li = document.querySelector(`.alerta[data-id="${datos.id}"]`);
      if (li && a) { li.classList.add("atendida"); li.querySelector(".acciones-alerta").replaceWith(accionesAlerta(a)); }
    }
  };
}

// ------------------------------------------------------------------ nueva cámara
let modoCamara = "enlace";
function elegirModo(modo) {
  modoCamara = modo;
  document.querySelectorAll(".pestana").forEach((b) => b.classList.toggle("activa", b.dataset.modo === modo));
  document.querySelectorAll(".panel-modo").forEach((p) => (p.hidden = p.dataset.panel !== modo));
}
document.querySelectorAll(".pestana").forEach((b) => (b.onclick = () => elegirModo(b.dataset.modo)));
const megas = (n) => `${(n / 1048576).toFixed(1)} MB`;
function mostrarArchivo() {
  const a = $("#archivo-video").files[0];
  $("#soltar-texto").innerHTML = a ? `<b></b><br><small>${megas(a.size)} · clic para cambiarlo</small>`
    : "Arrastre un video aquí o haga clic para elegirlo<br><small>MP4, AVI, MOV, MKV o WEBM · hasta 2 GB</small>";
  if (a) $("#soltar-texto b").textContent = a.name;
}
$("#archivo-video").onchange = mostrarArchivo;
const soltar = $("#soltar");
["dragenter", "dragover"].forEach((t) => soltar.addEventListener(t, (e) => { e.preventDefault(); soltar.classList.add("encima"); }));
["dragleave", "drop"].forEach((t) => soltar.addEventListener(t, () => soltar.classList.remove("encima")));
soltar.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length) { $("#archivo-video").files = e.dataTransfer.files; mostrarArchivo(); }
});

function subirVideo(archivo, titulo) {
  // se envía el archivo tal cual (sin formulario multiparte) para mostrar el avance de la subida
  return new Promise((ok, falla) => {
    const x = new XMLHttpRequest(), barra = $("#subida-progreso");
    x.open("PUT", `/api/videos?nombre=${encodeURIComponent(archivo.name)}&titulo=${encodeURIComponent(titulo)}`);
    barra.hidden = false; barra.value = 0;
    x.upload.onprogress = (e) => { if (e.lengthComputable) barra.value = (100 * e.loaded) / e.total; };
    x.onload = () => {
      barra.hidden = true;
      let r = {}; try { r = JSON.parse(x.responseText); } catch {}
      x.status < 300 ? ok(r) : falla(new Error(r.error || `Error ${x.status}`));
    };
    x.onerror = () => { barra.hidden = true; falla(new Error("Se cortó la conexión durante la subida.")); };
    x.send(archivo);
  });
}

async function esperarDescarga(id) {
  // los enlaces de YouTube, Facebook o TikTok se descargan en el servidor; se muestra el avance hasta que termina
  const caja = $("#descarga"), barra = $("#descarga-progreso"), txt = $("#descarga-texto");
  caja.hidden = false; barra.value = 0;
  try {
    for (;;) {
      await new Promise((ok) => setTimeout(ok, 1000));
      const d = await api(`/api/descargas/${id}`);
      if (d.estado === "error") throw new Error(`No se pudo descargar el video: ${d.error}`);
      barra.value = d.progreso;
      txt.textContent = d.estado === "descargando" ? `Descargando «${d.titulo || "video"}»… ${Math.round(d.progreso)} %`
        : d.estado === "lista" ? "Listo" : "Preparando la descarga…";
      if (d.estado === "lista") return d.camara;
    }
  } finally { caja.hidden = true; }
}

$("#btn-camara").onclick = () => { elegirModo(modoCamara); $("#dlg-camara").showModal(); };
$("#form-camara").onsubmit = async (e) => {
  e.preventDefault();
  const f = new FormData(e.target), boton = $("#btn-agregar");
  const nombre = (f.get("nombre") || "").trim();
  boton.disabled = true;
  try {
    if (modoCamara === "archivo") {
      const archivo = $("#archivo-video").files[0];
      if (!archivo) throw new Error("Elija un video de su PC.");
      boton.textContent = "Subiendo…";
      await subirVideo(archivo, nombre);
    } else {
      const fuente = (f.get("fuente") || "").trim();
      if (!fuente) throw new Error("Escriba el número de la webcam, la dirección RTSP o el enlace del video.");
      const r = await api("/api/camaras", { method: "POST", body: JSON.stringify({ nombre: nombre || fuente, fuente }) });
      if (r.descarga) { boton.textContent = "Descargando…"; await esperarDescarga(r.descarga); }
    }
    e.target.reset(); mostrarArchivo(); $("#dlg-camara").close();
  } catch (err) { alert(err.message); }
  finally { boton.disabled = false; boton.textContent = "Agregar"; }
};

// ------------------------------------------------------------------ ajustes
const leer = (o, ruta) => ruta.split(".").reduce((x, k) => x?.[k], o);
function escribir(o, ruta, v) { const k = ruta.split("."); const u = k.pop(); k.reduce((x, c) => (x[c] ??= {}), o)[u] = v; }
$("#btn-ajustes").onclick = async () => {
  const aj = await api("/api/ajustes");
  const cont = $("#clases-objeto");                 // las clases dependen del detector (COCO o vocabulario abierto)
  cont.replaceChildren(...Object.keys(aj.reglas.objetos.clases || {}).map((nombre) => {
    const l = el("label", "fila"), i = el("input");
    i.type = "checkbox"; i.dataset.ruta = `reglas.objetos.clases.${nombre}`;
    l.append(i, " " + nombre.charAt(0).toUpperCase() + nombre.slice(1));
    return l;
  }));
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
const COLORES = { restringida: "#e5534b", merodeo: "#e3a33a", protegida: "#3d7fd0" };
const franja = (z) => (z.horario ? ` · ${z.horario.desde}-${z.horario.hasta}` : " · todo el día");

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
    const li = el("li", null, `${z.nombre} · ${z.tipo}${franja(z)}`);
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
  const z = { nombre: $("#zona-nombre").value || "Zona", tipo: $("#zona-tipo").value, puntos };
  const desde = $("#zona-desde").value, hasta = $("#zona-hasta").value;
  if (desde && hasta && desde !== hasta) z.horario = { desde, hasta };   // vacío = activa todo el día
  zonas.push(z);
  puntos = []; $("#zona-nombre").value = `Zona ${zonas.length + 1}`;
  pintarListaZonas(); dibujarZonas();
};
$("#zona-deshacer").onclick = () => { puntos.pop(); dibujarZonas(); };
$("#zona-guardar").onclick = async () => {
  zonas = await api(`/api/camaras/${camZona}/zonas`, { method: "PUT", body: JSON.stringify(zonas) });
  $("#dlg-zonas").close();
};
addEventListener("resize", () => { if ($("#dlg-zonas").open) ajustarLienzo(); });

// ------------------------------------------------------------------ modo solo esqueleto
$("#btn-esqueleto").onclick = async () => {
  const b = $("#btn-esqueleto"), activo = b.getAttribute("aria-pressed") !== "true";
  b.setAttribute("aria-pressed", String(activo));
  await api("/api/ajustes", { method: "PUT", body: JSON.stringify({ privacidad: { solo_esqueleto: activo } }) });
};

// ------------------------------------------------------------------ borrar historial (entre pruebas o demostraciones)
$("#btn-limpiar").onclick = async () => {
  if (!confirm("¿Borrar todas las alertas y sus fotos?\n\nTambién se reinicia lo que VigIA aprendió de la escena " +
               "(objetos tomados como parte del lugar y falsas alarmas marcadas). Las cámaras, zonas y ajustes se conservan.")) return;
  try {
    const r = await api("/api/reiniciar", { method: "POST" });
    alertasCargadas.clear(); camaraEnAlerta.clear(); pintarAlertas();
    $("#btn-limpiar").title = `Última vez: ${r.borradas} alertas borradas a las ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    alert(`No se pudo borrar el historial: ${e.message || e}`);
  }
};

// ------------------------------------------------------------------ búsqueda con CLIP
function estadoBusqueda(d) {
  const txt = d.estado === "lista"
    ? `${d.total} capturas indexadas${d.pendientes ? `, ${d.pendientes} en cola` : ""}. Escriba lo que quiere encontrar, en español o en inglés.`
    : d.estado === "apagada" ? "La búsqueda está apagada: actívela en Ajustes → Búsqueda con CLIP."
    : `Búsqueda: ${d.estado}.`;
  $("#buscar-estado").textContent = txt;
}
$("#btn-buscar").onclick = async () => {
  $("#dlg-buscar").showModal(); $("#buscar-texto").focus();
  estadoBusqueda(await api("/api/buscar"));
};
$("#form-buscar").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("#buscar-texto").value.trim(); if (!q) return;
  const cont = $("#buscar-resultados"); cont.replaceChildren(el("p", "vacio", "Buscando…"));
  const r = await fetch(`/api/buscar?q=${encodeURIComponent(q)}&k=18`); const d = await r.json();
  estadoBusqueda(d);
  if (!r.ok) { cont.replaceChildren(el("p", "vacio", d.error)); return; }
  if (!d.resultados.length) { cont.replaceChildren(el("p", "vacio", "No hay capturas todavía.")); return; }
  const max = d.resultados[0].puntaje;
  cont.replaceChildren(...d.resultados.map((res) => {
    const a = res.alerta, t = el("button", "resultado");
    const img = el("img"); img.src = `/capturas/${res.captura}`; img.loading = "lazy"; img.alt = a ? a.mensaje : res.captura;
    t.append(img, el("b", null, a ? a.tipo_nombre : "Captura"), el("small", null, a ? `${a.camara} · ${fechaCorta(a.fecha)}` : res.captura));
    const barra = el("i", "similitud"); barra.style.width = `${Math.max(8, (100 * res.puntaje) / max)}%`;
    barra.title = `similitud ${res.puntaje.toFixed(3)}`; t.append(barra);
    if (a) t.onclick = () => abrirCaptura(a);
    return t;
  }));
};

// ------------------------------------------------------------------ inicio
const sel = $("#filtro-tipo");
Object.entries(TIPOS).forEach(([k, v]) => { const o = el("option", null, v); o.value = k; sel.append(o); });
sel.onchange = () => { filtro = sel.value; cargarAlertas(); pintarResumen(); };
cargarAlertas().catch(console.error);
conectar();
setInterval(pintarCamaras, 1000);
