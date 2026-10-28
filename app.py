
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import math
import os

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))


def reconocer_figuras(archivo):
    datos = np.frombuffer(archivo, np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)

    if imagen is None:
        return {"error": "No se pudo leer la imagen."}

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    _, binaria = cv2.threshold(
        gris, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    resultados = []

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < 150:
            continue

        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            continue

        aprox = cv2.approxPolyDP(
            contorno, 0.02 * perimetro, True
        )
        vertices = len(aprox)
        circularidad = 4 * math.pi * area / perimetro**2
        x, y, w, h = cv2.boundingRect(contorno)
        relacion = w / h if h else 0

        if circularidad > 0.82 and vertices >= 5:
            figura = "circulo"
        elif vertices >= 8 and circularidad < 0.65:
            figura = "estrella"
        elif vertices == 3:
            figura = "triangulo"
        elif vertices == 4:
            lados = [
                np.linalg.norm(
                    aprox[i][0] - aprox[(i + 1) % 4][0]
                )
                for i in range(4)
            ]
            es_cuadrado = (
                0.85 <= relacion <= 1.15
                and min(lados) > 0
                and max(lados) / min(lados) < 1.2
            )
            figura = "cuadrado" if es_cuadrado else "trapecio"
        else:
            figura = "null"

        resultados.append({
            "figura": figura,
            "vertices": vertices,
            "area": round(area, 2),
            "perimetro": round(perimetro, 2),
            "circularidad": round(circularidad, 3),
            "x": int(x),
            "y": int(y),
            "ancho": int(w),
            "alto": int(h)
        })

    return {"resultados": resultados}


@app.get("/")
def inicio():
    return send_file(os.path.join(BASE, "index.html"))


@app.post("/recognize")
def recognize():
    archivo = request.files.get("imagen")

    if not archivo:
        return jsonify({"error": "Selecciona una imagen."}), 400

    resultado = reconocer_figuras(archivo.read())
    return jsonify(resultado)


if __name__ == "__main__":
    app.run(debug=True)
