
```markdown
# 📷 EmotionCam — Informe Técnico y Documentación de Proyecto
**Asignatura:** Inteligencia Artificial Aplicada a Apps Móviles  
**Proyecto Final del Semestre** **Autores:** Yeison Betancur · Sara Dallos  

---

## 🛠️ 1. Componente: Ingeniería de Software

### Arquitectura de Software y Modularidad
El proyecto implementa una arquitectura desacoplada basada en el patrón de diseño sugerido en la rúbrica oficial, separando de manera estricta las vistas, el controlador de eventos y los servicios de Inteligencia Artificial. 

```text
[ Capa de Interfaz ]   <---> Cliente Web Móvil (HTML5, CSS3, JavaScript Asíncrono)
         ↓ ↑
[ Capa de Control ]    <---> Servidor REST Flask (app.py)
         ↓ ↑
[ Módulo Core IA ]     <---> Pipeline DeepFace (Detección ROI + Inferencia CNN)
         ↓
[ Capa Persistencia ]  <---> Base de Datos Relacional Local (SQLite3)

```

La lógica del servidor se modularizó mediante programación orientada a objetos y abstracción de servicios:

* **`app.py`:** Actúa como el controlador web central y API Gateway, administrando los endpoints HTTP POST/GET y gestionando el ciclo de vida de la aplicación.
* **`modules/database.py`:** Capa de acceso a datos (DAO) encargada de la persistencia relacional de las sesiones analíticas de forma aislada.
* **`modules/face_detector.py`:** Abstracción para el procesamiento morfológico inicial de imágenes.

### Persistencia de Datos

La aplicación garantiza el almacenamiento local de la telemetría a través de **SQLite3** (`data/emotioncam.db`). El esquema guarda de forma determinista:

1. ID único de sesión.
2. Marca de tiempo (Timestamp ISO).
3. Duración de la sesión en segundos.
4. Historial secuencial de la emoción dominante identificada y su confianza matemática decimal.

---

## 🧠 2. Componente: Inteligencia Artificial

### Marco de Inferencia y Pipeline de Visión

Para cumplir con los criterios de robustez y precisión exigidos en el curso, el núcleo cognitivo de la aplicación se migró hacia el ecosistema de **DeepFace**. Esta librería de vanguardia automatiza tareas de Visión por Computador complejas dentro de una sola tubería (*pipeline*):

1. **Segmentación y Detección del Rostro (Región de Interés - ROI):** El sistema analiza el flujo de la cámara en matrices de píxeles (BGR/RGB) y localiza las coordenadas espaciales de la cara.
2. **Inferencia mediante Redes Convolucionales Profundas (CNN):** La imagen recortada y normalizada se somete a una red convolucional especializada (arquitecturas densas tipo **VGG-Face**), entrenada con millones de imágenes faciales. La red extrae vectores de características morfológicas y los procesa a través de una función Softmax para clasificar la expresión en 7 categorías emocionales discretas: `happy`, `sad`, `neutral`, `angry`, `fear`, `surprise`, `disgust`.

### Optimización y Suavizado Dinámico (Prediction Smoothing)

Para resolver el problema del "ruido visual" o variaciones bruscas entre frames contiguos, el servidor implementa un filtro lineal basado en una estructura de datos de cola fija (`deque(maxlen=8)`). El algoritmo promedia y pondera los puntajes de las últimas 8 predicciones, devolviendo una **emoción estable** al frontend, mitigando parpadeos y falsos positivos causados por microexpresiones o cambios de iluminación.

---

## 📱 3. Componente: Diseño Móvil

### Enfoque Híbrido y Adaptabilidad (Responsive Design)

El proyecto implementa el enfoque de **Flask + App móvil híbrida**, utilizando estándares web modernos para emular una experiencia nativa. La interfaz se diseñó con CSS flexible (Flexbox y CSS Grid) asegurando que el panel analítico, el visualizador de la cámara y los botones de control se adapten dinámicamente a pantallas de dispositivos Android de cualquier resolución.

### Experiencia de Usuario (UX) y Comunicación Asíncrona

* **Captura de Medios Dinámica:** Utiliza el API nativo del navegador del teléfono (`navigator.mediaDevices.getUserMedia`) para abrir el hardware de la cámara frontal sin necesidad de plugins externos.
* **Procesamiento No Bloqueante:** JavaScript extrae los frames mediante un `canvas` oculto y los envía en ráfagas binarias comprimidas (`Form Data`) mediante peticiones `POST` asíncronas (`fetch`). Esto garantiza que la interfaz móvil se mantenga a **30 FPS estables**, sin congelar la pantalla del dispositivo mientras el computador ejecuta el procesamiento pesado del modelo de IA.

---

## 🔍 4. Componente: Explicación Técnica (Manual de Despliegue)

### Requisitos de Entorno

* Python 3.9 o superior.
* PC y Teléfono móvil conectados a la **misma red WiFi local**.

### Guía de Instalación del Servidor (PC)

```bash
# 1. Acceder al directorio raíz del repositorio clonado
cd EmotionCam

# 2. Inicializar el entorno virtual limpio
python -m venv .venv

# 3. Activar el entorno virtual (En Windows PowerShell)
.venv\Scripts\Activate.ps1

# 4. Instalar las dependencias de producción (Flask, OpenCV, DeepFace)
pip install -r requirements.txt

# 5. Ejecutar el servidor web seguro
python app.py

```

### Protocolo de Red Seguro (HTTPS) y Despliegue en Teléfono

Debido a las estrictas políticas de seguridad móvil, las cámaras de los celulares se bloquean en entornos HTTP comunes. Por ello, el servidor genera certificados SSL locales en tiempo real.

1. Al encender la app, tome nota de la IP local de su red (ej. `https://192.168.0.15:5000`).
2. **Instalación de la Entidad de Certificación (CA):** Desde el teléfono móvil, ingrese primero a `http://192.168.0.15:5001/emotioncam-ca.crt` para descargar el certificado raíz generado por su servidor.
3. Vaya a `Ajustes` -> `Seguridad` -> `Cifrado y credenciales` -> `Instalar desde almacenamiento` -> `Certificado CA` y cargue el archivo descargado.
4. Finalmente, abra el navegador del móvil e ingrese de forma segura a `https://192.168.0.15:5000` para operar el sistema en vivo.

```

```