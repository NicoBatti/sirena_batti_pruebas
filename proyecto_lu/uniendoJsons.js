// merge_inundaciones.js
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

let negativos = fs.readFileSync('registrosNegativos.json', 'utf8');
negativos = JSON.parse(negativos);
console.log("Registros negativos: "+ negativos.length);

let positivos = fs.readFileSync('inundaciones.json', 'utf8');
positivos = JSON.parse(positivos);
console.log("Registros positivos: "+ positivos.length);

function safeReadJSON(filePath) {
    try {
    const raw = fs.readFileSync(filePath, 'utf8');
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
        throw new Error(`El JSON en ${filePath} no es un array en el nivel raíz.`);
    }
    return parsed;
    } catch (err) {
    console.error(`Error leyendo o parseando ${filePath}:`, err.message);
    process.exit(1);
    }
}

// Intenta convertir la propiedad "Date (YMD)" a un timestamp para ordenar.
// Acepta formatos ISO (YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS...) y también "YYYYMMDD".
// Si no puede parsear, devuelve NaN, y esos registros se colocarán al final.
function parseDateYMD(value) {
    if (value == null) return NaN;
  // Si ya es número
    if (typeof value === 'number') return value;
  // Si es string, limpiar espacios
    let s = String(value).trim();

  // Caso común: YYYY-MM-DD o YYYY-MM-DDTHH:MM:SS
  // Date can parse ISO-like strings reliably.
  // If it's like YYYYMMDD, convert to YYYY-MM-DD.
  if (/^\d{8}$/.test(s)) {
    s = s.slice(0,4) + '-' + s.slice(4,6) + '-' + s.slice(6,8);
  }
  // Try Date.parse
  const t = Date.parse(s);
  if (!isNaN(t)) return t;
  // Fallback: try replace '/' with '-' (e.g., DD/MM/YYYY or YYYY/MM/DD)
  const s2 = s.replace(/\//g, '-');
  const t2 = Date.parse(s2);
  if (!isNaN(t2)) return t2;
  return NaN;
}

function addInundacionFlag(arr, flag) {
  return arr.map(item => {
    // clonamos el objeto para no mutar el original
    const copy = Object.assign({}, item);
    copy.inundacion = !!flag;
    return copy;
  });
}

function main() {
  const cwd = process.cwd();
  const fileA = path.join(cwd, 'inundaciones.json');
  const fileB = path.join(cwd, 'registrosNegativos.json');

  if (!fs.existsSync(fileA)) {
    console.error(`No se encontró ${fileA}`);
    process.exit(1);
  }
  if (!fs.existsSync(fileB)) {
    console.error(`No se encontró ${fileB}`);
    process.exit(1);
  }

  const a = safeReadJSON(fileA);
  const b = safeReadJSON(fileB);

  const aFlagged = addInundacionFlag(a, true);
  const bFlagged = addInundacionFlag(b, false);

  const combined = aFlagged.concat(bFlagged);

  // Ordenar: registros con fecha parseable primero (ascendente), luego los no-parseables.
  combined.sort((r1, r2) => {
    const d1 = parseDateYMD(r1['Date (YMD)']);
    const d2 = parseDateYMD(r2['Date (YMD)']);

    const n1 = isNaN(d1) ? Infinity : d1;
    const n2 = isNaN(d2) ? Infinity : d2;

    if (n1 < n2) return -1;
    if (n1 > n2) return 1;
    // Si las fechas son iguales o no parseables, mantener orden estable por si acaso:
    return 0;
  });

  const outPath = path.join(cwd, 'inundaciones_unidas.json');
  try {
    fs.writeFileSync(outPath, JSON.stringify(combined, null, 2), 'utf8');
    console.log(`Archivo escrito exitosamente: ${outPath}`);
    console.log(`Registros totales: ${combined.length}`);
  } catch (err) {
    console.error('Error escribiendo el archivo de salida:', err.message);
    process.exit(1);
  }
}

// Ejecutar main() solamente cuando este archivo sea el entrypoint ejecutado directamente.
const __filename = fileURLToPath(import.meta.url);
if (process.argv[1] === __filename) {
  main();
}