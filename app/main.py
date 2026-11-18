"""Local FastAPI application: uvicorn app.main:app --host 127.0.0.1 --workers 1."""

import asyncio
import tempfile
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartException

from app.api_schemas import JobDetailResponse, UploadMatchResponse
from app.api_service import MatchingService, ServiceUnavailable
from app.config import get_settings
from app.resume_parser import ResumeError
from app.schemas import Preferences


def problem(status, code, message):
    return HTTPException(status_code=status, detail={"code": code, "message": message})


class BodyLimit:
    """Bound multipart bytes before Starlette spools a large file to disk."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        settings = scope["app"].state.settings
        limit = settings.max_upload_mb * 1024 * 1024 + 65536
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = limit + 1
        async def reject():
            response = JSONResponse(status_code=413, content={"detail": {
                "code": "upload_too_large", "message": "Request exceeds the upload size limit."}})
            await response(scope, receive, send)
        if declared > limit:
            return await reject()
        size = 0
        exceeded = False
        replaced = False
        async def limited_receive():
            nonlocal size, exceeded
            message = await receive()
            size += len(message.get("body", b""))
            if size > limit:
                exceeded = True
                # The multipart parser closes its temporary files on this error.
                raise MultiPartException("Request exceeds upload limit")
            return message
        async def limited_send(message):
            nonlocal replaced
            if exceeded:
                if not replaced:
                    replaced = True
                    await reject()
                return
            await send(message)
        try:
            await self.app(scope, limited_receive, limited_send)
        except MultiPartException:
            if not replaced:
                await reject()


def create_app(settings=None, service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.settings = settings or get_settings()
        app.state.service = service or MatchingService(app.state.settings)
        app.state.gate = asyncio.Semaphore(1)
        app.state.runs = OrderedDict()
        app.state.tasks = set()
        yield
        if app.state.tasks:
            await asyncio.gather(*app.state.tasks, return_exceptions=True)
        app.state.runs.clear()

    app = FastAPI(title="JobFit — Synthetic Job Matcher", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse(status_code=422, content={"detail": {
            "code": "invalid_request", "message": "Check the required fields and their types."}})

    def prune():
        now = time.monotonic()
        for key, (expires, _) in list(app.state.runs.items()):
            if expires <= now:
                app.state.runs.pop(key, None)

    def get_run(run_id):
        prune()
        if run_id not in app.state.runs:
            raise problem(404, "run_not_found", "Run is unknown or expired; upload the resume again.")
        return app.state.runs[run_id][1]

    async def execute(function, *args, cleanup=None):
        if app.state.gate.locked():
            if cleanup:
                cleanup()
            raise problem(429, "busy", "Another request is processing; retry shortly.")
        await app.state.gate.acquire()
        def work():
            try:
                return function(*args)
            finally:
                if cleanup:
                    cleanup()
        task = asyncio.create_task(run_in_threadpool(work))
        app.state.tasks.add(task)
        def finished(future):
            app.state.tasks.discard(future)
            app.state.gate.release()
            if not future.cancelled():
                future.exception()  # consume errors after a timeout/disconnect
        task.add_done_callback(finished)
        try:
            return await asyncio.wait_for(asyncio.shield(task), app.state.settings.api_timeout_seconds)
        except (asyncio.TimeoutError, TimeoutError):
            raise problem(504, "processing_timeout", "Processing exceeded the request deadline; retry later.")
        except ResumeError as error:
            raise problem(422, error.code, str(error))
        except ServiceUnavailable:
            raise problem(503, "service_unavailable", "Prepared corpus, model, or vector index is unavailable; check setup.")
        except KeyError:
            raise problem(404, "job_not_found", "Job does not exist in this corpus.")
        except Exception:
            raise problem(500, "processing_failed", "Processing failed; no resume content was logged.")

    @app.get("/health")
    async def health():
        prune()
        return {"status": "ok", "is_synthetic": True, "busy": app.state.gate.locked()}

    @app.get("/ready")
    async def ready():
        if not app.state.service.ready():
            raise problem(503, "corpus_unavailable", "Run Module 1 dataset preparation first.")
        return {"status": "ready", "is_synthetic": True,
                "note": "Corpus is loaded; model and vector index are verified when used."}

    @app.post("/match", response_model=UploadMatchResponse)
    async def match(resume: UploadFile = File(...), preferences: str = Form("{}"), use_llm: bool = Form(True)):
        path = None
        handed_off = False
        try:
            if len(preferences.encode()) > 16384:
                raise problem(422, "invalid_preferences", "Preferences exceed the allowed size.")
            try:
                prefs = Preferences.model_validate_json(preferences)
            except ValidationError:
                raise problem(422, "invalid_preferences", "Preferences must be valid JSON matching the preferences schema.")
            if not app.state.service.ready():
                raise problem(503, "corpus_unavailable", "Run Module 1 dataset preparation first.")
            if app.state.gate.locked():
                raise problem(429, "busy", "Another request is processing; retry shortly.")
            folder = app.state.settings.artifacts_dir / "uploads"
            folder.mkdir(parents=True, exist_ok=True)
            limit = app.state.settings.max_upload_mb * 1024 * 1024
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".pdf", delete=False) as target:
                path = Path(target.name)
                size = 0
                while chunk := await resume.read(65536):
                    size += len(chunk)
                    if size > limit:
                        raise problem(413, "upload_too_large", "Resume exceeds the configured size limit.")
                    target.write(chunk)
            deadline = time.perf_counter() + app.state.settings.api_timeout_seconds
            handed_off = True
            run = await execute(app.state.service.match, path, prefs, use_llm, deadline,
                                cleanup=lambda: path.unlink(missing_ok=True))
            prune()
            while len(app.state.runs) >= app.state.settings.api_max_sessions:
                app.state.runs.popitem(last=False)
            run_id = str(uuid4())
            ttl = app.state.settings.api_session_ttl_seconds
            app.state.runs[run_id] = (time.monotonic() + ttl, run)
            return UploadMatchResponse(run_id=run_id, expires_in_seconds=ttl,
                                       result=run.response, preparation_warnings=run.prepared.warnings)
        except OSError:
            raise problem(503, "upload_storage_unavailable", "Temporary upload storage is unavailable.")
        finally:
            await resume.close()
            if path is not None and not handed_off:
                path.unlink(missing_ok=True)

    @app.get("/runs/{run_id}/jobs/{job_id}", response_model=JobDetailResponse)
    async def detail(run_id: str, job_id: str):
        run = get_run(run_id)
        return await execute(app.state.service.detail, run, job_id)

    @app.delete("/runs/{run_id}", status_code=204)
    async def forget(run_id: str):
        get_run(run_id)
        app.state.runs.pop(run_id, None)

    return app


app = create_app()
