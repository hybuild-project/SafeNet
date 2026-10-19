# SafeNet

SafeNet analizza immagini con un modello ONNX e restituisce le aree rilevate, il punteggio e i riquadri. La libreria Python offre anche la detect dei riquadri; la demo browser visualizza i riquadri senza inviare immagini a un server.

## Installazione Python

Richiede Python 3.8 o superiore. Dalla radice del repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Su Windows attivare l’ambiente con `.venv\Scripts\activate`.

## Rilevamento

```python
from safenet import SafeDetector

detector = SafeDetector()
detections = detector.detect("immagine.jpg")
batch = detector.detect_batch(["uno.jpg", "due.jpg"])
```

`detect` accetta un percorso, un array OpenCV, bytes o un file binario aperto. `detect_batch` restituisce una lista di rilevamenti per ciascuna immagine. Ogni rilevamento contiene `class`, `score` e `box` nel formato `[x, y, larghezza, altezza]`.

Le etichette pubbliche, mantenute nello stesso ordine degli output del modello, sono:

```text
F_GENIE_CROSSED, FAUCET_F, BUTTONS_EXPRESS, F_BEST_EXPRESS,
F_GENIE_EXPRESS, M_BEST_EXPRESS, ANGUS_EXPRESS, FIT_EXPRESS,
BELL_CROSSED, FIT_CROSSED, ARMCHAIRS_CROSSED, ARMCHAIRS_EXPRESS,
FAUCET_M, BELL_EXPRESS, M_GENIE_EXPRESS, ANGUS_CROSSED,
F_BEST_CROSSED, BUTTONS_CROSSED
```

`320n.onnx` è il modello predefinito: SafeNet lo cerca prima in `models/320n.onnx`, poi nel package. Per selezionare un altro ONNX basta passare il percorso; risoluzione e classi vengono lette dai metadati del modello:

```python
detector = SafeDetector(model_path="models/f_best-v4.2.onnx")
predictions = detector.detect("immagine.jpg")
```

Sono utilizzabili allo stesso modo `640m.onnx`, `f_best_v4.2.onnx` e `top_yolov8s.onnx`. Per `320n` e `640m` le classi vengono restituite con gli alias SafeNet; gli altri modelli usano i nomi dichiarati nei metadati ONNX. È possibile forzare la risoluzione con `inference_resolution=640` oppure con una coppia `(larghezza, altezza)`, ma normalmente non serve. I checkpoint `.pt` non sono caricati direttamente: esportarli prima in ONNX.

## GPU

SafeDetector sceglie automaticamente CUDA quando ONNX Runtime rileva `CUDAExecutionProvider`, altrimenti usa la CPU. Se l’inizializzazione CUDA fallisce, ripiega sulla CPU. Per forzare i provider, passarli al costruttore, per esempio `SafeDetector(providers=["CPUExecutionProvider"])`.

Installare una sola distribuzione ONNX Runtime compatibile con il sistema:

```bash
# CPU
python -m pip install onnxruntime

# GPU NVIDIA: sostituire la versione CPU con quella GPU e verificare driver/CUDA supportati
python -m pip uninstall -y onnxruntime
python -m pip install onnxruntime-gpu
```

Verificare i provider disponibili con `python -c "import onnxruntime as o; print(o.get_available_providers())"`. La presenza del provider non garantisce da sola che driver e librerie CUDA installate siano compatibili.

## MagicDetector desktop

Installare le dipendenze GUI e di scansione, poi avviare dalla radice del repository:

```bash
python -m pip install -r MagicDetector/requirements.txt
python -m MagicDetector.MagicDetector
```

Il comando va eseguito dalla radice del repository, non da `in_browser`. Su Ubuntu/Debian installa le librerie native Qt/XCB con:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libegl1 libxkbcommon0 libxkbcommon-x11-0 \
	libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
	libxcb-render-util0 libxcb-shape0 libxcb-util1 libxcb-xkb1
```

Serve anche una sessione grafica accessibile: installare questi pacchetti non crea un `DISPLAY` in un container headless.

La finestra permette di scegliere al volo uno degli ONNX in `models/`, aggiungere cartelle da scansionare, escludere cartelle e consultare la galleria indicizzata. Il modello scelto viene usato dalla scansione successiva; non viene cambiato nel mezzo di una scansione già avviata. I risultati e le preferenze sono salvati nei database locali `images_index.db` e `settings.db` nella directory di lavoro. Il database immagini creato dalla versione precedente usa chiavi diverse: esegui una nuova scansione in un database vuoto dopo l’aggiornamento.

## Demo in browser

La demo React esegue la rilevazione localmente nel browser. Richiede Node.js e npm:

```bash
cd in_browser
npm install
npm start
```

Il menu della demo permette di cambiare modello senza ricaricare la pagina; i pesi vengono caricati al momento della selezione. `npm run build` copia gli ONNX da `../models/` nel sito statico, quindi il build contiene circa 240 MB con l’attuale set di modelli. La demo disegna i riquadri e le etichette sull’immagine selezionata; non applica una censura permanente all’immagine.

## FastDeploy

`fastdeploy_recipe/` contiene un predictor FastDeploy senza interfaccia grafica: `predictor(image_paths, batch_size=1, model_name="320n.onnx")` accetta il nome di un ONNX in `models/` e carica il modello alla prima richiesta o quando cambia `model_name`. Il container include gli ONNX presenti in `models/` (non i checkpoint `.pt`). Per eseguirlo, installare prima SafeNet dalla radice (`python -m pip install -e .`), quindi installare le dipendenze FastDeploy del recipe e avviare il servizio secondo la versione di FastDeploy in uso.

## Addestramento

`train.py` prepara i dati e avvia Ultralytics YOLO11 con `yolo11n.pt` per impostazione predefinita. In un ambiente Python dedicato installare PyTorch appropriato per il sistema (CPU o CUDA), poi:

```bash
python -m pip install ultralytics Pillow requests tqdm
python train.py
```

Modificare `YOLO11_MODEL` e gli altri parametri nella sezione di configurazione per scegliere modello, batch e augmentations. Per esportare i checkpoint `.pt` presenti in `models/` in ONNX eseguire `python pt2onnx.py`; lo script ignora gli ONNX già esportati. `train.py` registra in memoria le estensioni temporanee del dataset, senza modificare l’installazione Ultralytics. Scarica un dataset e usa un checkpoint iniziale pubblicati esternamente; il checkpoint può richiedere il download manuale dal link nel sorgente. Il processo può richiedere molto spazio, tempo e memoria GPU. Prima di avviare un training completo, verificare dataset, annotazioni, mapping delle classi e compatibilità dell’export ONNX. Il training non esporta né sostituisce automaticamente `safenet/320n.onnx`.

## Note sul modello

SafeNet restituisce categorie e riquadri, non una decisione infallibile: verificare i risultati prima di usare le immagini in flussi automatici.
