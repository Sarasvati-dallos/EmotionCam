# EmotionCam
**Detector de emociones faciales en tiempo real**
Yeison Betancur · Sara Dallos — Proyecto IA

---

## ¿Cómo funciona?
La app corre en tu PC y te conectas desde el teléfono Android por WiFi.
El teléfono actúa como pantalla, la PC procesa la cámara y el modelo de IA.

---

## Instalación

```bash
# 1. Entrar a la carpeta
cd EmotionCamFlask

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Poner el modelo de Yeison en la carpeta models/
#    models/emotion_model.h5
#    models/emotion_labels.json   ← ya está incluido

# 5. Correr la app
python app.py
```

---

## Abrir en el teléfono
1. Conecta el teléfono al **mismo WiFi** que la PC
2. La terminal mostrará algo como:
   ```
   Teléfono: http://192.168.1.X:5000
   ```
3. Abre esa dirección en el navegador del teléfono
4. ¡Listo!

---

## Estructura
```
EmotionCamFlask/
├── app.py                  # Servidor Flask principal
├── requirements.txt
├── models/
│   ├── emotion_model.h5    # ← de Yeison
│   └── emotion_labels.json
├── modules/
│   ├── emotion_classifier.py
│   ├── face_detector.py
│   └── database.py
├── templates/
│   ├── index.html          # Inicio + Cámara
│   ├── history.html        # Historial
│   └── stats.html          # Estadísticas
└── data/
    └── emotioncam.db       # Se crea automáticamente
```
