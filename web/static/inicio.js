"use strict";
// Escena de la portada: personas representadas solo por su esqueleto (17 puntos COCO) caminan frente a una
// cámara alta; una se cae y VigIA la marca con la medida que disparó la alerta, como en el panel real.

const C = {
  niebla: "#e6ecef", tinta: "#12202b", pizarra: "#56656f", esqueleto: "#0d7f74", alerta: "#c93a26", rejilla: "#c7d1d7",
};
const FUENTE = '"Bahnschrift", "DIN Alternate", "Barlow", "Roboto Condensed", "Arial Narrow", sans-serif';
const UNIONES = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
const CICLO = 9.5;                      // segundos de la historia de la caída
const quieto = matchMedia("(prefers-reduced-motion: reduce)").matches;

const lienzo = document.getElementById("escena");
const ctx = lienzo.getContext("2d");
let W = 0, H = 0, dpr = 1, horizonte = 0, compacto = false;   // compacto: celular, la escena va sobre el texto

/** 17 puntos de una persona de altura 1 con los pies en (0, 0); y crece hacia abajo, como en la imagen. */
function postura(fase, d, caminando) {
  const s = caminando ? Math.sin(fase) : 0, k = [];
  const fijo = { 0: [0.02 * d, 0.93], 1: [0.035 * d, 0.95], 2: [0, 0.95], 3: [0.05 * d, 0.94], 4: [-0.02 * d, 0.94],
    5: [0.11, 0.80], 6: [-0.11, 0.80], 11: [0.07, 0.50], 12: [-0.07, 0.50] };
  for (const [i, [x, y]] of Object.entries(fijo)) k[i] = [x, -y];
  [[5, 7, 9, 1], [6, 8, 10, -1]].forEach(([h, c, m, sg]) => {
    const osc = 0.10 * s * sg;
    k[c] = [k[h][0] + 0.03 * sg + osc * 0.6, -0.64];
    k[m] = [k[h][0] + 0.03 * sg + osc, -0.48];
  });
  [[11, 13, 15, 1], [12, 14, 16, -1]].forEach(([c, r, t, sg]) => {
    const paso = 0.12 * s * sg;
    k[r] = [k[c][0] + paso * 0.5, -0.27];
    k[t] = [k[c][0] + paso, 0];
  });
  return k;
}

/** Gira el cuerpo alrededor de los pies (la caída). */
function girar(k, ang) {
  const c = Math.cos(ang), s = Math.sin(ang);
  return k.map(([x, y]) => [x * c - y * s, Math.min(0, x * s + y * c)]);
}

// profundidad z: 0 lejos, 1 cerca. Las personas caminan en la mitad superior para dejar libre el texto.
const pieY = (z) => horizonte + (H - horizonte) * (compacto ? 0.22 + 0.7 * z : 0.16 + 0.34 * z);
const altura = (z) => H * (compacto ? 0.17 + 0.3 * z : 0.11 + 0.17 * z);

const caminantes = [
  { id: 3, z: 0.10, x: 0.18, v: 0.55, d: 1 }, { id: 7, z: 0.30, x: 0.82, v: 0.62, d: -1 },
  { id: 9, z: 0.48, x: 0.35, v: 0.50, d: 1 }, { id: 14, z: 0.22, x: 0.60, v: 0.70, d: -1 },
  { id: 21, z: 0.62, x: 0.05, v: 0.58, d: 1 },
].map((p) => ({ ...p, fase: Math.random() * 6 }));

function piso() {
  ctx.fillStyle = C.niebla; ctx.fillRect(0, 0, W, H);
  const vx = W * 0.5;
  ctx.strokeStyle = C.rejilla; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = -14; i <= 14; i++) {              // líneas que fugan hacia el horizonte
    ctx.moveTo(vx + (i * W) / 7, H); ctx.lineTo(vx + (i * W) / 60, horizonte);
  }
  for (let j = 1; j <= 14; j++) {                // líneas horizontales cada vez más juntas a lo lejos
    const y = horizonte + (H - horizonte) * Math.pow(j / 14, 2.1);
    ctx.moveTo(0, y); ctx.lineTo(W, y);
  }
  ctx.stroke();
}
function velo() {
  if (compacto) return;                          // en el celular el texto va debajo de la escena
  // detrás del titular y del texto, las personas se desvanecen para que se lea
  const g = ctx.createLinearGradient(0, H * 0.5, 0, H);
  g.addColorStop(0, "rgba(230,236,239,0)"); g.addColorStop(0.5, "rgba(230,236,239,.9)"); g.addColorStop(1, C.niebla);
  ctx.fillStyle = g; ctx.fillRect(0, H * 0.5, W, H * 0.5);
  const l = ctx.createLinearGradient(0, 0, W * 0.56, 0);
  l.addColorStop(0, "rgba(230,236,239,.86)"); l.addColorStop(0.75, "rgba(230,236,239,.6)"); l.addColorStop(1, "rgba(230,236,239,0)");
  ctx.fillStyle = l; ctx.fillRect(0, horizonte * 0.6, W * 0.56, H);
}

function persona(k, x, y, h, color, id, grosor = 1, caja = true) {
  const pts = k.map(([a, b]) => [x + a * h, y + b * h]);
  ctx.strokeStyle = color; ctx.lineCap = "round"; ctx.lineWidth = Math.max(1.5, h * 0.016) * grosor;
  ctx.beginPath();
  for (const [a, b] of UNIONES) { if (a > 4 || b > 4) { ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); } }
  // la cabeza: un círculo sobre los cinco puntos de la cara (nariz, ojos y orejas)
  const cx = pts.slice(0, 5).reduce((s, p) => s + p[0], 0) / 5, cy = pts.slice(0, 5).reduce((s, p) => s + p[1], 0) / 5;
  ctx.moveTo(cx + h * 0.065, cy); ctx.arc(cx, cy, h * 0.065, 0, 7);
  ctx.moveTo(...pts[0]); ctx.lineTo((pts[5][0] + pts[6][0]) / 2, (pts[5][1] + pts[6][1]) / 2);
  ctx.stroke();
  ctx.fillStyle = color;
  for (const [px, py] of pts.slice(5)) { ctx.beginPath(); ctx.arc(px, py, Math.max(1.6, h * 0.012), 0, 7); ctx.fill(); }
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]), m = h * 0.1;     // margen que incluye la cabeza
  const r = [Math.min(...xs) - m, Math.min(...ys) - m * 1.6, Math.max(...xs) + m, Math.max(...ys) + m * 0.4];
  if (caja) {
    ctx.lineWidth = grosor > 1 ? 2.5 : 1; ctx.strokeRect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
    ctx.font = `500 ${Math.round(11 + h * 0.04)}px ${FUENTE}`;
    ctx.fillText(`#${id}`, r[0], r[1] - 6);
  }
  return r;
}

function etiqueta(r, alfa) {
  const fs = Math.max(13, Math.min(17, W / 85)), pad = 12, ancho = Math.min(360, W * (compacto ? 0.8 : 0.42));
  const lineas = partir("El tronco quedó a 84° de la vertical y la cadera bajó 0.43 alturas en 0.8 s.", ancho - pad * 2, fs);
  const alto = pad * 2 + fs * 1.3 + lineas.length * fs * 1.35 + 4;
  // encima de la caja, alineada a su borde derecho, sin salirse de la escena
  let x = Math.max(12, Math.min(W - ancho - 12, r[2] - ancho)), y = r[1] - alto - 30;
  if (y < horizonte * 0.6) y = r[3] + 12;
  ctx.globalAlpha = alfa;
  ctx.fillStyle = "rgba(255,255,255,.94)"; ctx.fillRect(x, y, ancho, alto);
  ctx.fillStyle = C.alerta; ctx.fillRect(x, y, 4, alto);
  ctx.fillStyle = C.alerta; ctx.font = `600 ${fs * 1.12}px ${FUENTE}`;
  ctx.fillText("Caída detectada", x + pad + 4, y + pad + fs);
  ctx.fillStyle = C.tinta; ctx.font = `400 ${fs}px ${FUENTE}`;
  lineas.forEach((l, i) => ctx.fillText(l, x + pad + 4, y + pad + fs * 1.3 + (i + 1) * fs * 1.35));
  ctx.globalAlpha = 1;
}
function partir(txt, ancho, fs) {
  ctx.font = `400 ${fs}px ${FUENTE}`;
  const out = []; let linea = "";
  for (const p of txt.split(" ")) {
    const prueba = linea ? `${linea} ${p}` : p;
    if (ctx.measureText(prueba).width > ancho && linea) { out.push(linea); linea = p; } else linea = prueba;
  }
  return [...out, linea];
}

function rotulo(t) {
  const fs = Math.max(12, Math.min(15, W / 100));
  ctx.font = `400 ${fs}px ${FUENTE}`; ctx.fillStyle = C.pizarra; ctx.textAlign = "right";
  const hora = new Date().toLocaleTimeString("es-PE", { hour12: false });
  ctx.fillText(`Cámara 1   ${hora}   ${Math.round(9.6 + Math.sin(t) * 0.4)} cuadros/s`, W - Math.max(16, W * 0.04),
    compacto ? H - 12 : Math.max(84, H * 0.1));
  ctx.textAlign = "left";
}

function cuadro(t, dt) {
  piso();
  rotulo(t);
  const orden = [...caminantes].sort((a, b) => a.z - b.z);
  for (const p of orden) {
    const h = altura(p.z);
    if (dt) {
      p.x += (p.d * p.v * h * dt) / W;
      p.fase += (p.v * dt * 2 * Math.PI) / 0.55;
      if (p.x > 1.1) p.x = -0.1; if (p.x < -0.1) p.x = 1.1;
    }
    persona(postura(p.fase, p.d, true), p.x * W, pieY(p.z), h, C.esqueleto, p.id);
  }
  velo();
  // la protagonista: camina, se cae, VigIA la detecta y explica por qué
  const c = t % CICLO, z = 0.78, h = altura(z), y = pieY(z);
  const x0 = W * (compacto ? 0.3 : 0.58), x1 = W * (compacto ? 0.5 : 0.74);
  const avance = Math.min(1, c / 3.2), x = x0 + (x1 - x0) * avance;
  const cae = Math.min(1, Math.max(0, (c - 3.2) / 0.8)), levanta = Math.min(1, Math.max(0, (c - 7.6) / 0.9));
  const f = Math.max(0, cae - levanta), suave = f * f * (3 - 2 * f);
  const ang = (Math.PI / 180) * 86 * suave;
  const detectada = c > 3.7 && c < 8.2;
  let k = postura(c * 2 * Math.PI * 0.95, 1, c < 3.2);
  if (ang > 0) k = girar(k, ang);
  const alfa = c > 8.6 ? Math.max(0, 1 - (c - 8.6) / 0.6) : Math.min(1, c / 0.4);
  ctx.globalAlpha = alfa;
  const r = persona(k, x, y, h, detectada ? C.alerta : C.esqueleto, 12, detectada ? 1.6 : 1);
  ctx.globalAlpha = 1;
  if (detectada) etiqueta(r, Math.min(1, (c - 3.7) / 0.3) * Math.min(1, (8.2 - c) / 0.3));
}

function medir() {
  const caja = lienzo.getBoundingClientRect();
  dpr = Math.min(2, devicePixelRatio || 1);
  W = caja.width; H = caja.height; compacto = W < 760; horizonte = H * (compacto ? 0.2 : 0.13);
  lienzo.width = Math.round(W * dpr); lienzo.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

let t = 0, previo = null, visible = true, pedido = null;
function bucle(ahora) {
  const dt = previo == null ? 0 : Math.min(0.05, (ahora - previo) / 1000);
  previo = ahora; t += dt;
  cuadro(t, dt);
  pedido = visible ? requestAnimationFrame(bucle) : null;
}
function arrancar() { if (!pedido && !quieto) { previo = null; pedido = requestAnimationFrame(bucle); } }

medir();
if (quieto) cuadro(5.2, 0);          // un solo cuadro con la caída ya detectada
else arrancar();
addEventListener("resize", () => { medir(); if (quieto) cuadro(5.2, 0); });
new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible) arrancar(); }).observe(lienzo);
document.addEventListener("visibilitychange", () => { visible = !document.hidden; if (visible) arrancar(); });

// ------------------------------------------------------------------ arranque (una vez por sesión)
// El sistema «se enciende»: se dibuja el ojo, se revisa cada parte con datos reales del servidor y el ojo se
// abre sobre la escena. Cualquier tecla o toque lo salta.
async function arranque() {
  let visto = true;
  try { visto = !!sessionStorage.getItem("vigia_arranque"); sessionStorage.setItem("vigia_arranque", "1"); } catch {}
  if (visto || quieto) return;
  const capa = document.getElementById("arranque"), lista = document.getElementById("arranque-lineas");
  capa.hidden = false;
  document.documentElement.style.overflow = "hidden";
  let saltar = false, abierto = false;
  const abrir = () => {
    if (abierto) return; abierto = true;
    document.getElementById("portada").classList.add("iris");
    capa.classList.add("abriendo");
    setTimeout(() => { capa.remove(); document.documentElement.style.overflow = ""; }, 1000);
  };
  const salto = () => { saltar = true; abrir(); };
  addEventListener("keydown", salto, { once: true });
  capa.addEventListener("pointerdown", salto, { once: true });

  const pedir = (u) => Promise.race([fetch(u).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    new Promise((ok) => setTimeout(() => ok(null), 900))]);
  const [estado, ajustes] = await Promise.all([pedir("/api/estado"), pedir("/api/ajustes")]);
  const enLinea = estado ? estado.camaras.filter((c) => c.conectada).length : null;
  const priv = ajustes?.privacidad || {};
  const lineas = [
    ["Modelo de pose YOLO11n: 17 puntos por persona", true],
    ["Seguimiento de personas entre cuadros", true],
    ["13 reglas de comportamiento", true],
    enLinea == null ? ["Cámaras: sin respuesta del servidor", false]
      : enLinea ? [`${enLinea} ${enLinea === 1 ? "cámara en línea" : "cámaras en línea"}`, true]
      : ["Sin cámaras conectadas: agrégalas en el panel", false],
    estado?.solo_esqueleto ? ["Privacidad: solo esqueleto", true]
      : priv.difuminar_rostros === false ? ["Privacidad: pixelado de rostros apagado", false]
      : ["Privacidad: rostros pixelados", true],
  ];
  await new Promise((ok) => setTimeout(ok, 900));          // mientras se dibuja el ojo
  for (const [texto, ok] of lineas) {
    if (saltar) return;
    const li = document.createElement("li"), t = document.createElement("span"), m = document.createElement("span");
    m.className = ok ? "marca-ok" : "marca-aviso";
    li.append(t, m); lista.append(li);
    for (let i = 1; i <= texto.length; i++) {               // se escribe letra por letra
      if (saltar) return;
      t.textContent = texto.slice(0, i);
      await new Promise((r) => setTimeout(r, 11));
    }
    m.textContent = ok ? "listo" : "revisar";
    await new Promise((r) => setTimeout(r, 110));
  }
  await new Promise((r) => setTimeout(r, 450));
  abrir();
}
arranque();

// estado en vivo del sistema (solo si el servidor responde)
async function enVivo() {
  try {
    const r = await fetch("/api/estado"); if (!r.ok) return;
    const d = await r.json();
    const camaras = d.camaras.filter((c) => c.conectada).length;
    const alertas = Object.values(d.resumen || {}).reduce((a, x) => a + x.total, 0);
    const p = document.getElementById("en-vivo");
    p.replaceChildren();
    const dato = (n, txt, punto) => {
      const s = document.createElement("span"), b = document.createElement("b");
      b.textContent = n; s.append(b);
      if (punto) { const i = document.createElement("i"); i.className = "punto"; s.append(i); }
      s.append(txt); p.append(s);
    };
    dato(camaras, camaras === 1 ? "cámara en línea" : "cámaras en línea", camaras > 0);
    dato(alertas, alertas === 1 ? "alerta hoy" : "alertas hoy", false);
    p.hidden = false;
  } catch {}
}
enVivo(); setInterval(enVivo, 5000);
