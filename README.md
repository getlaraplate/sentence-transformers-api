# sentence-api

Piccolo servizio HTTP (Flask) che espone embeddings tramite
[sentence-transformers](https://www.sbert.net/). Usato dallo stack Laraplate:
il client invia in ogni richiesta il `service_model` del profilo attivo, quindi
il modello di default serve solo quando la richiesta non specifica `model`.

## Requisiti

- Python 3.11+ (`requirements.txt` fissa `numpy==2.3.2`, che richiede 3.11)
- Le dipendenze in [`requirements.txt`](requirements.txt)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configurazione (variabili d'ambiente)

| Variabile                | Default                              | Descrizione                                   |
| ------------------------ | ------------------------------------ | --------------------------------------------- |
| `EMBEDDING_MODEL`        | `intfloat/multilingual-e5-small`     | Modello caricato all'avvio / usato di default |
| `EMBEDDING_MODEL_CACHE`  | `2`                                  | Quanti modelli tenere residenti (LRU)         |
| `EMBEDDING_PORT`         | `8000`                               | Porta di ascolto                              |
| `CROSS_ENCODER_MODEL`    | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Modello di rerank di default per `/score`, caricato all'avvio. Stringa vuota: rerank spento, `/score` risponde 503 se la richiesta non indica `model` |
| `CROSS_ENCODER_MODEL_CACHE` | `1`                               | Quanti modelli di rerank tenere residenti (LRU) |
| `CROSS_ENCODER_MAX_LENGTH`  | `512`                             | Token massimi per coppia (il resto viene troncato) |
| `CROSS_ENCODER_MAX_PAIRS`   | `64`                              | Coppie massime per richiesta (il client Laraplate ne invia al massimo 64) |

I modelli vengono scaricati automaticamente da Hugging Face al primo utilizzo e
non sono versionati nel repo.

## Avvio

```bash
python sentence-api.py
```

Il servizio ascolta su `0.0.0.0` sulla porta `EMBEDDING_PORT` (default `8000`).

Il modello di default viene caricato all'avvio: se non è scaricabile o
utilizzabile il processo termina subito, invece di partire e rispondere 500.

## API

### `GET /health`

Stato del servizio e modelli caricati.

```json
{
  "status": "healthy",
  "model": "intfloat/multilingual-e5-small",
  "default_model": "intfloat/multilingual-e5-small",
  "loaded_models": ["intfloat/multilingual-e5-small"],
  "model_cache": 2,
  "reranker": {
    "default_model": null,
    "loaded_models": [],
    "model_cache": 1,
    "max_length": 512,
    "max_pairs": 64
  }
}
```

`reranker.default_model` a `null` indica un host che non fa rerank.

### `GET /models`

Endpoint di discovery: il servizio carica qualsiasi modello su richiesta, quindi
non c'è un elenco fisso di modelli ammessi. Restituisce il default, i modelli
attualmente residenti e le famiglie che supportano `input_type`, così il client
sa quando `input_type` conta.

```json
{
  "default_model": "intfloat/multilingual-e5-small",
  "loaded_models": ["intfloat/multilingual-e5-small"],
  "model_cache": 2,
  "input_types": ["query", "passage"],
  "prefix_families": {
    "e5": { "query": "query: ", "passage": "passage: " },
    "nomic": { "query": "search_query: ", "passage": "search_document: " }
  }
}
```

### `POST /embed`

Genera gli embeddings. Body JSON:

- `text` (string) **oppure** `texts` (array di string) — obbligatorio
- `model` (string, opzionale) — override del modello di default
- `input_type` (`"query"` | `"passage"`, opzionale) — intento **semantico**:
  `query` = sto cercando, `passage` = sto indicizzando. Il server garantisce il
  prefisso corretto per il modello (es. e5 → `query:` / `passage:`) in modo
  **idempotente**: se il testo è già prefissato lo lascia, altrimenti lo
  aggiunge (nessun rischio di `query: query: ...`). Per i modelli senza
  convenzione viene ignorato. Se omesso, nessun prefisso.
- `normalize_embeddings` (bool, opzionale, default `true`)

Il client resta **agnostico rispetto al modello**: manda sempre la stessa forma
e lascia al server la conoscenza di come si parla a ciascun modello.

```bash
curl -s http://localhost:8000/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts": ["ciao mondo", "hello world"], "input_type": "passage"}'
```

Risposta:

```json
{
  "model": "intfloat/multilingual-e5-small",
  "input_type": "passage",
  "prefix_applied": true,
  "embeddings": [[...], [...]]
}
```

### `POST /score`

Rerank con un cross-encoder: dà un punteggio di pertinenza a ogni coppia
domanda-documento. Body JSON:

- `pairs` (array di `{"query": string, "text": string}`) — obbligatorio, al
  massimo `CROSS_ENCODER_MAX_PAIRS`
- `model` (string, opzionale) — override del modello di default

Il modello deve essere un **cross-encoder** (un modello addestrato per il rerank,
non un modello di embeddings) e, per contenuti in italiano, multilingue. Il
server passa la sigmoide in modo esplicito, quindi i punteggi sono sempre in
`[0, 1]` anche per i checkpoint che di default restituiscono logit.

```bash
curl -s http://localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"pairs": [{"query": "orari di apertura", "text": "Il museo è aperto dalle 9 alle 18"}]}'
```

Risposta:

```json
{ "model": "<modello>", "scores": [0.93] }
```

Errori: `400` per un body malformato o troppe coppie, `503` se non c'è nessun
modello (`CROSS_ENCODER_MODEL` vuoto e nessun `model` nella richiesta), `500` se il
modello non restituisce un punteggio per coppia. Il client Laraplate tratta
qualunque errore come "rerank non eseguito" e tiene l'ordine originale.

Dal lato Laraplate l'URL di base si imposta con `CROSS_ENCODER_URL` (es.
`http://HOST:8000`, senza `/score`, che aggiunge il client). Se non è impostato si
usa `SENTENCE_TRANSFORMERS_URL`: il rerank gira sullo stesso servizio degli
embeddings. Non c'è un indirizzo predefinito. La chiave è `CROSS_ENCODER_API_KEY`,
con ripiego su `SENTENCE_TRANSFORMERS_API_KEY`.

## Test

```bash
python -m unittest discover -s tests
```

I test sostituiscono `sentence-transformers` e `torch` con finti: non scaricano
nessun modello e verificano cosa il servizio chiede al modello e cosa risponde.

## Deploy (systemd)

Il file [`sentence-transformers.service`](sentence-transformers.service) è la
unit usata in produzione. Presuppone il virtualenv in `/opt/ai-env` e il codice
in `/opt/sentence-api`. Il rerank è attivo con il modello di default. Per configurarlo aggiungere alla unit
`Environment=CROSS_ENCODER_MODEL=<modello>` per cambiare il modello di rerank, o
`Environment=CROSS_ENCODER_MODEL=` per spegnerlo, prima di riavviare.

```bash
sudo cp sentence-transformers.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sentence-transformers.service
```

## Licenza

Rilasciato sotto [Apache License 2.0](LICENSE).
