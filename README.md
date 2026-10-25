# CAA con Show-o2 1.5B

[Abrir en Colab](https://colab.research.google.com/github/Nicerova7/show-o2-space/blob/experiment/caa-understanding/colab-caa-understanding.ipynb)

El piloto CAA agrega dos archivos a la estructura original del repo:

- `colab-caa-understanding.ipynb`: setup, carga, tus 20 pares, tokens seleccionados, resta y promedio, imagen + prompt y tres coeficientes.
- `experiments/caa_utils.py`: carga de Show-o2, preparación de imagen y hooks de extracción y steering.

Usa GPU con BF16 (A100 o L4) y High-RAM. Ejecuta setup, reinicia la sesión y continúa con la carga. Pega tus 20 tripletas en `BUILD_EXAMPLES`: `(pregunta, respuesta positiva, respuesta negativa)`.

La extracción es textual, como en tu notebook. Cada par sólo cambia la letra final A/B. Qwen fusiona `(A` en un token: usamos una letra suelta para extraer exactamente A/B. La auditoría muestra los tokens antes de ejecutar el modelo.

El vector es `mean(h_positivo - h_negativo)`, sin normalización adicional. Capturamos e intervenimos la salida del bloque elegido, antes de la norma final. La capa 12 es un punto de partida editable.

La evaluación recibe imagen + prompt y produce texto con `c = -2, 0, 2`. El hook suma `c*v` desde el último token del input hasta el final. El sampler original recalcula toda la secuencia sin KV cache: el rango crece de 1 a 2 a 3 posiciones. El notebook muestra esa traza y permite descargar los resultados.

La imagen se redimensiona y recorta al centro a 432×432; el recorte se muestra. Show-o se descarga por separado al commit `45a5a2de01d1ebd10cd5864d29310a76476cdf23`.

Rama `experiment/caa-understanding`, creada desde `main`. Verificados: sintaxis, esquema del notebook, letras con el tokenizer real de Qwen y aritmética/rangos del hook con tensores pequeños. Inferencia completa con pesos en GPU pendiente de ejecución en Colab.

Referencias: [paper CAA](https://arxiv.org/html/2312.06681v4#S3), [repo CAA](https://github.com/nrimsky/CAA), [Show-o2](https://github.com/showlab/Show-o/tree/45a5a2de01d1ebd10cd5864d29310a76476cdf23/show-o2).

La estructura original de `main` se conserva: `show_o2_space/`, `inputs/`, `outputs/`, `experiments/` y `colab.ipynb`. El piloto sigue la organización de `experiment/text-red-steering`: un notebook propio y sus utilidades en `experiments/`.
