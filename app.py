import os
import mimetypes
import secrets
import hmac
import json
import contextlib
import anyio
from pathlib import Path
from typing import Optional, AsyncIterator

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Depends, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# MCP imports
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

# ---------- config ----------
STORAGE_ROOT = Path(os.environ.get("STORAGE_ROOT", "./public_storage")).resolve()
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

MAX_PREVIEW    = 2 * 1024 * 1024
MAX_UPLOAD     = int(os.environ.get("MAX_UPLOAD_MB", "0")) * 1024 * 1024  # 0 = unlimited
MAX_DEPTH      = 20
AUTH_USER      = os.environ.get("STORAGE_USER", "jarvis")
AUTH_PASS      = os.environ.get("STORAGE_PASS", "stark2024")
SESSION_TOKEN  = secrets.token_urlsafe(32)
CHUNK          = 1024 * 1024  # 1 MB

# ---------- FastAPI app ----------
app = FastAPI(title="J.A.R.V.I.S. Storage + MCP")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ============================================================
# Shared helpers
# ============================================================
def safe_path(rel: str) -> Path:
    rel = (rel or "").strip().lstrip("/\\")
    target = (STORAGE_ROOT / rel).resolve()
    try:
        target.relative_to(STORAGE_ROOT)
    except ValueError:
        raise HTTPException(400, "Path escapes storage root")
    return target


def _walk(p: Path, remaining: int):
    if remaining <= 0:
        return []
    children = []
    try:
        entries = sorted(
            (e for e in p.iterdir() if not e.is_symlink()),
            key=lambda x: (not x.is_dir(), x.name.lower()),
        )
    except (PermissionError, OSError):
        return []
    for entry in entries:
        try:
            st = entry.stat()
        except OSError:
            continue
        node = {
            "name": entry.name,
            "path": entry.relative_to(STORAGE_ROOT).as_posix(),
            "is_dir": entry.is_dir(),
            "size": st.st_size if entry.is_file() else None,
            "modified": st.st_mtime,
        }
        if entry.is_dir():
            node["children"] = _walk(entry, remaining - 1)
        children.append(node)
    return children


def require_auth(request: Request):
    tok = request.cookies.get("jarvis_auth")
    if not tok or not hmac.compare_digest(tok, SESSION_TOKEN):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")


# ============================================================
# REST — auth
# ============================================================
class LoginReq(BaseModel):
    user: str
    pass_: str


@app.post("/api/login")
def login(body: LoginReq):
    user_ok = hmac.compare_digest(body.user.encode(), AUTH_USER.encode())
    pass_ok = hmac.compare_digest(body.pass_.encode(), AUTH_PASS.encode())
    if not (user_ok and pass_ok):
        raise HTTPException(401, "Invalid credentials")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        "jarvis_auth", SESSION_TOKEN,
        httponly=True, samesite="lax",
        max_age=60 * 60 * 24 * 7,
        path="/",
    )
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("jarvis_auth", path="/")
    return resp


# ============================================================
# REST — storage
# ============================================================
@app.get("/api/tree", dependencies=[Depends(require_auth)])
def tree(path: str = "", depth: int = MAX_DEPTH):
    root = safe_path(path)
    if not root.exists():
        raise HTTPException(404, "Not found")
    if not root.is_dir():
        raise HTTPException(400, "Not a directory")
    depth = max(1, min(depth, MAX_DEPTH))
    return {"root": STORAGE_ROOT.name, "path": path, "children": _walk(root, depth)}


@app.get("/api/preview", dependencies=[Depends(require_auth)])
async def preview(path: str):
    p = safe_path(path)
    if not p.is_file():
        raise HTTPException(404, "Not a file")
    size = p.stat().st_size
    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "application/octet-stream"

    if size > MAX_PREVIEW:
        return {"type": "large", "size": size, "mime": mime}
    if mime.startswith("image/"):
        return {"type": "image", "url": f"/api/download?path={path}", "mime": mime}
    if mime.startswith("text/") or mime in (
        "application/json", "application/xml", "application/x-yaml",
        "application/javascript",
    ):
        try:
            text = await anyio.to_thread.run_sync(
                lambda: p.read_text(encoding="utf-8", errors="replace")
            )
        except Exception as e:
            return {"type": "error", "message": str(e)}
        return {"type": "text", "content": text, "mime": mime, "size": size}
    return {"type": "binary", "mime": mime, "size": size,
            "url": f"/api/download?path={path}"}


@app.get("/api/download", dependencies=[Depends(require_auth)])
def download(path: str, request: Request):
    p = safe_path(path)
    if not p.is_file():
        raise HTTPException(404, "Not a file")

    file_size = p.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        try:
            _, rng = range_header.split("=", 1)
            start_s, end_s = (rng.split("-", 1) + [""])[:2]
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
            end = min(end, file_size - 1)
            if start > end:
                raise ValueError
        except Exception:
            raise HTTPException(416, "Invalid Range")

        length = end - start + 1

        def iter_range():
            with open(p, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iter_range(), status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Content-Disposition": f'attachment; filename="{p.name}"',
                "Cache-Control": "no-store",
            },
            media_type="application/octet-stream",
        )

    def iter_all():
        with open(p, "rb") as f:
            while True:
                chunk = f.read(CHUNK)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        iter_all(),
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Disposition": f'attachment; filename="{p.name}"',
            "Cache-Control": "no-store",
        },
        media_type="application/octet-stream",
    )


@app.post("/api/upload", dependencies=[Depends(require_auth)])
async def upload(path: str = "", file: UploadFile = File(...)):
    target_dir = safe_path(path)
    target_dir.mkdir(parents=True, exist_ok=True)

    raw = (file.filename or "").replace("\\", "/").split("/")[-1].strip()
    if not raw or raw in (".", ".."):
        raise HTTPException(400, "Invalid filename")
    dest = (target_dir / raw).resolve()
    if dest.parent != target_dir:
        raise HTTPException(400, "Invalid filename")

    tmp = dest.with_suffix(dest.suffix + ".part")
    total = 0
    try:
        with open(tmp, "wb") as f:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if MAX_UPLOAD and total > MAX_UPLOAD:
                    raise HTTPException(413, f"File exceeds {MAX_UPLOAD // (1024*1024)} MB")
                await anyio.to_thread.run_sync(f.write, chunk)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try: tmp.unlink()
            except OSError: pass

    return {"ok": True, "path": dest.relative_to(STORAGE_ROOT).as_posix(), "size": total}


class DeleteReq(BaseModel):
    path: str


@app.post("/api/delete", dependencies=[Depends(require_auth)])
def delete(req: DeleteReq):
    p = safe_path(req.path)
    if not p.exists():
        raise HTTPException(404, "Not found")
    if p.is_dir():
        raise HTTPException(400, "Refusing to delete a directory")
    try:
        p.unlink()
    except FileNotFoundError:
        raise HTTPException(404, "Not found")
    except OSError as e:
        raise HTTPException(500, f"Delete failed: {e}")
    return {"ok": True}


# ============================================================
# MCP SERVER — tools that talk to storage directly
# ============================================================
mcp = FastMCP("jarvis-storage")


@mcp.tool()
def list_files(path: str = "") -> list[dict]:
    """List files and folders under a storage path (recursive, depth 20)."""
    root = safe_path(path)
    if not root.exists():
        raise ValueError("Not found")
    if not root.is_dir():
        raise ValueError("Not a directory")
    return _walk(root, MAX_DEPTH)


@mcp.tool()
def read_file(path: str, max_bytes: int = 200_000) -> dict:
    """Return the text content of a file. Truncates at max_bytes."""
    p = safe_path(path)
    if not p.is_file():
        raise ValueError("Not a file")
    size = p.stat().st_size
    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "application/octet-stream"

    is_text = (
        mime.startswith("text/")
        or mime in ("application/json", "application/xml",
                    "application/x-yaml", "application/javascript")
    )
    if not is_text:
        return {"path": path, "type": "binary", "mime": mime, "size": size}
    if size > MAX_PREVIEW:
        return {"path": path, "type": "large", "size": size, "mime": mime}

    content = p.read_text(encoding="utf-8", errors="replace")
    if len(content) > max_bytes:
        content = content[:max_bytes] + "\n…[truncated]"
    return {"path": path, "type": "text", "content": content,
            "mime": mime, "size": size}


@mcp.tool()
def write_file(path: str, content: str, overwrite: bool = True) -> dict:
    """Create or overwrite a text file at the given storage path."""
    dest = safe_path(path)
    if dest.exists() and not overwrite:
        raise ValueError("File exists (overwrite=False)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try: tmp.unlink()
            except OSError: pass
    return {"ok": True, "path": dest.relative_to(STORAGE_ROOT).as_posix(),
            "size": dest.stat().st_size}


@mcp.tool()
def delete_file(path: str) -> dict:
    """Delete a single file (never a directory)."""
    p = safe_path(path)
    if not p.exists():
        raise ValueError("Not found")
    if p.is_dir():
        raise ValueError("Refusing to delete a directory")
    p.unlink()
    return {"ok": True, "path": path}


@mcp.tool()
def make_dir(path: str) -> dict:
    """Create a directory (and parents)."""
    p = safe_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "path": path}


@mcp.resource("storage://{path}")
def storage_resource(path: str) -> str:
    """Expose any text file in storage as an MCP resource."""
    p = safe_path(path)
    if not p.is_file():
        raise ValueError("Not a file")
    return p.read_text(encoding="utf-8", errors="replace")


# ============================================================
# Mount MCP at /mcp (Streamable HTTP)
# ============================================================
session_manager = StreamableHTTPSessionManager(
    app=mcp._mcp_server,
    json_response=False,
    stateless=True,
)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with session_manager.run():
        yield


app.router.lifespan_context = lifespan


@app.api_route("/mcp", methods=["GET", "POST", "DELETE"])
async def mcp_endpoint(request: Request):
    """MCP Streamable HTTP endpoint."""
    return await session_manager.handle_request(request.scope, request.receive, request._send)


# ============================================================
# Static UI (must be mounted LAST)
# ============================================================
UI_DIR = Path(__file__).parent / "static"
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    import uvicorn
    size_mb = MAX_UPLOAD // (1024 * 1024) if MAX_UPLOAD else "unlimited"
    print(f"Serving storage: {STORAGE_ROOT}")
    print(f"Max upload: {size_mb} MB")
    print(f"Login: {AUTH_USER}")
    print(f"REST:     http://localhost:8000/api/*")
    print(f"UI:       http://localhost:8000/")
    print(f"MCP:      http://localhost:8000/mcp")
    uvicorn.run(app, host="0.0.0.0", port=8000)
