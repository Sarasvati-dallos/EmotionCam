# EmotionCam - Contexto tecnico para otra IA

Este archivo resume el funcionamiento actual del proyecto y sus limitaciones conocidas. La idea es servir como contexto para una IA futura que vaya a preparar una presentacion o una explicacion de la app.

## 1. Que es el proyecto

EmotionCam es una app de deteccion de emociones faciales en tiempo real. El flujo actual ya no depende de la camara del PC como fuente principal: la idea es que el telefono abra la interfaz web, use la camara desde el navegador y envie frames al servidor Flask que corre en la PC.

La aplicacion corre en la PC con Flask y Flask-SocketIO. El telefono se conecta por WiFi desde el navegador, captura imagenes con getUserMedia y las envia por POST al backend.

## 2. Flujo actual de funcionamiento

### 2.1 Inicio del servidor

- El punto de entrada es `app.py`.
- Al arrancar, el servidor detecta la IP local de salida a red para mostrar la direccion correcta al telefono.
- Tambien genera o reutiliza certificados HTTPS locales.
- Se levanta un servidor HTTPS principal en el puerto 5000.
- Adicionalmente se levanta un servidor HTTP en el puerto 5001 para descargar la CA local.

### 2.2 Acceso desde el telefono

- El telefono debe estar en la misma red WiFi que la PC.
- La app web debe abrirse en `https://<IP_DE_LA_PC>:5000`.
- Para que el navegador permita la camara, el contexto debe ser seguro.
- Por eso se usa una CA local descargable desde `http://<IP_DE_LA_PC>:5001/emotioncam-ca.crt`.
- El usuario instala esa CA en Android para que el certificado HTTPS del servidor deje de marcar error.

### 2.3 Captura y envio de frames

- La interfaz web usa `navigator.mediaDevices.getUserMedia()`.
- El video se muestra en el telefono.
- Cada cierto intervalo, el cliente captura un frame en canvas y lo envia a `/api/frame`.
- El frame puede viajar como multipart form-data o como base64 JSON.

### 2.4 Procesamiento en servidor

- `/api/frame` decodifica la imagen recibida con OpenCV.
- Se detectan rostros con `FaceDetector`.
- Cada rostro se recorta y pasa al clasificador de emociones.
- El resultado se devuelve como JSON y tambien alimenta el estado de la UI.
- Si hay sesion activa, se guardan detecciones en SQLite.

### 2.5 Clasificacion de emociones

- El clasificador vive en `modules/emotion_classifier.py`.
- Carga `models/emotion_model.h5` y `models/emotion_labels.json`.
- Si el modelo no se puede cargar, cae a un clasificador placeholder aleatorio.
- El preprocesamiento actual intenta seguir el formato de entrenamiento: imagen en escala de grises por contenido, pero adaptada al formato que espera la base MobileNet del modelo.
- Si el modelo espera 3 canales, la imagen gris se replica a RGB antes de aplicar `mobilenet_v2.preprocess_input()`.
- Si el modelo espera 1 canal, la imagen se deja como gris y se reescala a una unica canal.

## 3. Archivos principales

### `app.py`

Responsable de:

- inicializar Flask y SocketIO,
- servir la pagina principal y rutas auxiliares,
- crear certificados HTTPS locales,
- exponer `/api/frame`, `/api/start`, `/api/stop`, `/api/status`, `/api/debug`, `/api/reset-smoothing`,
- manejar el flujo de deteccion y guardado en base de datos,
- emitir el estado actual de emocion.

### `modules/emotion_classifier.py`

Responsable de:

- cargar el modelo H5,
- cargar las etiquetas,
- preprocesar la cara antes de inferencia,
- convertir la salida del modelo en emocion + confianza + emoji.

### `templates/index.html`

Responsable de:

- interfaz principal tipo PWA,
- acceso a camara,
- envio periodico de frames al backend,
- mostrar emocion, confianza y estado de conexion,
- enlace para descargar la CA local.

### `modules/face_detector.py`

Responsable de:

- detectar caras en el frame,
- recortar la region facial para inferencia.

### `modules/database.py`

Responsable de:

- crear y manejar la base SQLite,
- guardar sesiones,
- guardar detecciones,
- calcular estadisticas e historiales.

## 4. Estado actual de la inferencia

Hay dos ideas importantes:

1. El entrenamiento original uso imagenes en blanco y negro, pero el pipeline de entrenamiento puede haber adaptado esas imagenes al formato de MobileNet.
2. Por eso la inferencia no debe convertir a color arbitrariamente. Debe replicar el mismo preprocesamiento del entrenamiento.

En la implementacion actual:

- se conserva el contenido grayscale cuando corresponde,
- se adapta la dimensionalidad al tipo de entrada que pide el modelo,
- se usa `mobilenet_v2.preprocess_input()` para mantener la misma escala de entrenamiento.

## 5. Mecanismos de estabilidad

La salida de emociones no se toma tal cual frame por frame. El backend usa un suavizado temporal para evitar parpadeo constante entre clases.

Actualmente existe:

- una ventana de predicciones recientes,
- una puntuacion EMA por clase,
- una logica para no cambiar de emocion por un solo frame aislado,
- un pequeno sesgo de seguridad para reducir dominancia excesiva de una clase si aparece muy frecuentemente.

Tambien existe un endpoint de depuracion:

- `GET /api/debug` para revisar la ventana y los scores,
- `POST /api/reset-smoothing` para reiniciar el estado de suavizado.

## 6. Limitaciones conocidas

### 6.1 Dependencia de HTTPS y CA

- En Android moderno, la camara del navegador requiere contexto seguro.
- Si el certificado no esta confiado, el navegador puede bloquear la camara o advertir error.
- El flujo depende de que el usuario instale la CA local.

### 6.2 Dependencia de red local

- PC y telefono deben estar en la misma red.
- Si la red aísla dispositivos o bloquea trafico local, el telefono no podra enviar frames.
- En Windows, el firewall puede requerir una regla manual para abrir el puerto 5000.

### 6.3 Calidad de deteccion

- La precision depende de la calidad del rostro, iluminacion, angulo y distancia.
- Si la cara sale parcialmente oculta, la deteccion puede fallar o generar confianza baja.
- El sesgo del modelo puede hacer que algunas expresiones se confundan con otras, sobre todo si la distribucion de entrenamiento fue desbalanceada.

### 6.4 Latencia

- El telefono captura y envia frames por intervalos, no en tiempo real absoluto.
- La latencia depende del WiFi, del navegador, del procesamiento OpenCV y de TensorFlow.
- El suavizado mejora estabilidad pero introduce un pequeno retraso de respuesta.

### 6.5 Dependencia del archivo del modelo

- Si `models/emotion_model.h5` o `models/emotion_labels.json` faltan, el sistema cae a placeholder y las predicciones pasan a ser aleatorias.
- Esto permite que la app siga funcionando, pero no con resultados reales.

### 6.6 Coste de TensorFlow

- TensorFlow puede tardar en arrancar y consumir bastante memoria.
- En Windows aparece el mensaje de oneDNN y warnings de compilacion del modelo cargado; no suelen ser un error fatal, pero alargan el inicio.

### 6.7 Posible desalineacion entre entrenamiento e inferencia

- Si el modelo fue entrenado con un preprocesamiento ligeramente distinto al actual, puede haber sesgos o confusiones.
- Los puntos mas criticos a comparar son:
  - tamano de entrada,
  - tipo de normalizacion,
  - uso real de 1 canal o 3 canales,
  - orden de las etiquetas.

## 7. Recomendaciones para la IA que hara la presentacion

- Explicar que el telefono no procesa la emocion: solo captura y envia frames.
- Explicar que la PC es la que corre el modelo de IA y la base de datos.
- Mencionar que la camara del navegador necesita HTTPS, por eso se usa CA local.
- No decir que el entrenamiento fue RGB si la fuente real fue grayscale con expansion para MobileNet.
- Si se habla de exactitud del modelo, aclarar que hay suavizado temporal y que los resultados no son por frame aislado.
- Si se menciona la arquitectura, separar claramente frontend PWA, backend Flask, clasificador, detector facial y base de datos.

## 8. Resumen corto

La app actual funciona como una PWA web: el telefono captura video, el backend Flask en la PC recibe frames por HTTPS, detecta rostros, clasifica emociones con el modelo H5 y devuelve un estado estable gracias a suavizado temporal. El sistema depende de certificados locales, red WiFi compartida y de que el archivo del modelo exista en `models/`.
