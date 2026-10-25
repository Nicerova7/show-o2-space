# CAA con Show-o2 1.5B

[Abrir en Colab](https://colab.research.google.com/github/Nicerova7/show-o2-space/blob/experiment/caa-understanding/colab-caa-understanding.ipynb)

El piloto CAA agrega dos archivos a la estructura original del repo:

- `colab-caa-understanding.ipynb`: setup, carga, 100 pares oficiales de CAA, tokens seleccionados, resta y promedio, imagen + prompt y tres coeficientes.
- `experiments/caa_utils.py`: funciones para construir los pares y hooks de extracción y steering.

El Colab importa `ShowO2Adapter` y `MODEL_ID` de `show_o2_space.adapter` para cargar
el modelo, preparar la imagen y generar texto. Importa las funciones CAA de
`experiments.caa_utils`, siguiendo la estructura de `text-red-steering`.

Usa GPU con BF16 (A100 o L4) y High-RAM. Ejecuta setup, reinicia la sesión y continúa con la carga.

El setup descarga el repo CAA. El Colab lee
`/content/CAA/datasets/generate/sycophancy/generate_dataset.json`
y selecciona 100 ejemplos sin reemplazo con `random.Random(SEED).sample`.
Conserva preguntas, opciones y etiquetas oficiales; no reordena A/B.
Guarda los ejemplos seleccionados en `selected_examples.json` dentro del ZIP.
Haz push de los cambios locales antes de usar el enlace de Colab.

La extracción es textual. La positiva usa `answer_matching_behavior` y la negativa
`answer_not_matching_behavior`. Cada par sólo cambia la letra final A/B.
Se usa el formato de chat de Qwen en lugar de `tokenize_llama_chat`.
Qwen fusiona `(A` en un token: representamos las etiquetas oficiales `(A)` / `(B)`
como letras sueltas y extraemos en `-1`, exactamente A/B. El `-2` del ejemplo Llama
no se traslada literalmente a Qwen. La auditoría muestra los tokens antes de ejecutar el modelo.

El vector es `mean(h_positivo - h_negativo)`, sin normalización adicional. Capturamos e intervenimos la salida del bloque elegido, antes de la norma final. La capa 12 es un punto de partida editable.

La evaluación recibe imagen + prompt y produce texto con `c = -2, 0, 2`. El hook suma `c*v` desde el último token del input hasta el final. El sampler original recalcula toda la secuencia sin KV cache: el rango crece de 1 a 2 a 3 posiciones. El notebook muestra esa traza y permite descargar los resultados.

La imagen se redimensiona y recorta al centro a 432×432; el recorte se muestra. Show-o se descarga por separado al commit `45a5a2de01d1ebd10cd5864d29310a76476cdf23`.

Rama `experiment/caa-understanding`, creada desde `main`. Verificados: sintaxis, esquema del notebook, letras con el tokenizer real de Qwen y aritmética/rangos del hook con tensores pequeños. Inferencia completa con pesos en GPU pendiente de ejecución en Colab.

Referencias: [paper CAA](https://arxiv.org/html/2312.06681v4#S3), [repo CAA](https://github.com/nrimsky/CAA), [Show-o2](https://github.com/showlab/Show-o/tree/45a5a2de01d1ebd10cd5864d29310a76476cdf23/show-o2).

La estructura original de `main` se conserva: `show_o2_space/`, `inputs/`, `outputs/`, `experiments/` y `colab.ipynb`. El piloto sigue la organización de `experiment/text-red-steering`: un notebook propio y sus utilidades en `experiments/`.
