from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.app_config import AppConfig
from app.db import connect_db, fetch_recent_predictions, init_db, log_prediction
from app.data.preprocessing import load_vocabs, tokenize_and_pad
from app.model.predictor import DisasterTwittsPredictor
from app.schemas import PredictRequest, PredictResponse, PredictionLog
from app.utils.errors import AppError, ArtifactError
from app.utils.seed import seed_everything
from app.utils.validation import InputValidator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = AppConfig()
    seed_everything(config.SEED)

    if not config.VOCAB_PATH.exists():
        raise ArtifactError(
            "VOCAB_NOT_FOUND",
            "Vocabulary file not found",
            {"path": str(config.VOCAB_PATH)},
        )
    
    if not config.MODEL_PATH.exists():
        raise ArtifactError(
            "MODEL_NOT_FOUND",
            "Model file not found",
            {"path": str(config.MODEL_PATH)}
        )
    
    word2idx, idx2word, vocab_size = load_vocabs(config.VOCAB_PATH)
    predictor = DisasterTwittsPredictor.from_config(config, vocab_size)
    if config.LABEL_MAPPING_PATH.exists():
        predictor.load_label_mapping(config.LABEL_MAPPING_PATH)
    validator = InputValidator(config)
    db_conn = connect_db(config.DB_PATH)
    init_db(db_conn)

    app.state.config = config
    app.state.word2idx = word2idx
    app.state.predictor = predictor
    app.state.validator = validator
    app.state.db = db_conn

    yield

    try:
        db_conn.close()
    except Exception:
        logger.exception("Failed to close DB connection")

app = FastAPI(title="Disaster Tweet Predictor", lifespan=lifespan)
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


def run_prediction(tweet: str, keyword: str | None = None) -> PredictResponse:
    config = app.state.config
    word2idx = app.state.word2idx
    predictor = app.state.predictor
    validator = app.state.validator
    db_conn = app.state.db

    final_text, warnings = validator.validate(tweet, keyword=keyword)
    input_ids, input_length = tokenize_and_pad(final_text, word2idx, config)
    prob, label, label_name = predictor.predict(
        input_ids, input_length, return_label_name=True
    )

    response = PredictResponse(
        probability=prob,
        label=label,
        label_name=label_name,
        threshold=predictor.threshold,
        warnings=list(warnings) if warnings else [],
    )

    try:
        log_prediction(
            db_conn,
            tweet=tweet,
            keyword=keyword,
            final_text=final_text,
            probability=prob,
            label=label,
            label_name=label_name,
            threshold=predictor.threshold,
            warnings=list(warnings) if warnings else [],
        )
    except Exception:
        logger.exception("Failed to log prediction")

    return response

@app.exception_handler(AppError)
def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(status_code=400, content=exc.to_dict())

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "error": None,
            "form_data": {"tweet": "", "keyword": ""},
        },
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return run_prediction(tweet=req.tweet, keyword=req.keyword)


@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    request: Request, tweet: str = Form(...), keyword: str = Form(default="")
):
    keyword_value = keyword.strip() or None
    try:
        result = run_prediction(tweet=tweet, keyword=keyword_value)
        error = None
    except AppError as exc:
        result = None
        error = exc.to_dict()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result.model_dump() if result else None,
            "error": error,
            "form_data": {"tweet": tweet, "keyword": keyword},
        },
    )


@app.get("/logs", response_model=list[PredictionLog])
def logs(limit: int = 50):
    db_conn = app.state.db
    limit = max(1, min(limit, 200))
    return fetch_recent_predictions(db_conn, limit=limit)


