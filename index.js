import fs from "fs";
import fetch from "node-fetch"; // Asegúrate de tener 'node-fetch' instalado

async function main() {
    try {
        // Leer el archivo JSON de forma síncrona
        let alturas = fs.readFileSync("alturasHidrometricas.json", 'utf8');
        alturas = JSON.parse(alturas);
        
        // Empezamos a escribir el array JSON en el archivo
        fs.writeFileSync("alturaRios.json", "[\n", 'utf8');
        
        // Iterar sobre cada objeto en el array 'alturas'
        for (let i = 0; i < alturas.length; i++) {
            const seriesId = alturas[i].seriesid;
            const siteCode = alturas[i].sitecode;
            
            try {
                // Hacer la petición a la API con los datos de cada objeto
                const respuesta = await fetch(`https://alerta.ina.gob.ar/pub/datos/datos&timeStart=1960-01-01&timeEnd=2025-09-20&seriesId=${seriesId}&siteCode=${siteCode}&varId=2&format=json`);

                // Verificar si la respuesta fue exitosa
                if (!respuesta.ok) {
                    throw new Error(`Error de red: ${respuesta.status}`);
                }

                // Parsear la respuesta como JSON
                const datosJson = await respuesta.json();
                
                // Convertir el JSON a string para escribirlo
                const datosString = JSON.stringify(datosJson, null, 2);

                // Agregar el JSON al archivo. Usamos una coma si no es el último elemento.
                fs.appendFileSync("alturaRios.json", datosString + (i < alturas.length - 1 ? ",\n" : "\n"), 'utf8');

                console.log(`Datos para seriesId ${seriesId} guardados.`);
                
            } catch (error) {
                console.error(`Error al obtener datos para seriesId ${seriesId}:`, error);
                // Si la petición falla, igual agregamos un JSON vacío para mantener la estructura del array.
                fs.appendFileSync("alturaRios.json", "{}\n", 'utf8');
            }
        }
        
        // Cerramos el array JSON al final del archivo
        fs.appendFileSync("alturaRios.json", "]", 'utf8');
        
        console.log("Archivo alturaRios.json creado exitosamente. 🎉");

    } catch (error) {
        console.error("Error general:", error);
    }
}

// Ejecutar la función principal
main();