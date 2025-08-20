import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Path, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

# Note: Using a simple in-memory repository by default, with optional JSON file persistence.
# Configuration is done via environment variables. No values are hardcoded.
# To configure persistence, set:
# - NOTES_DB_URL=file://path/to/notes.json  -> to persist notes into a JSON file
# Otherwise, the app will run purely in-memory for the lifecycle of the process.

# Settings and configuration


class Settings(BaseModel):
    """Application settings loaded from environment variables."""
    app_name: str = Field(default=os.getenv("APP_NAME", "Simple Notes API"))
    environment: str = Field(default=os.getenv("ENVIRONMENT", "development"))
    # Optional JSON file persistence via file URL e.g. file:///data/notes.json or file://notes.json
    notes_db_url: Optional[str] = Field(default=os.getenv("NOTES_DB_URL"))
    # CORS
    allow_origins: List[str] = Field(
        default_factory=lambda: os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
        if os.getenv("CORS_ALLOW_ORIGINS") else ["*"]
    )
    allow_methods: List[str] = Field(
        default_factory=lambda: os.getenv("CORS_ALLOW_METHODS", "*").split(",")
        if os.getenv("CORS_ALLOW_METHODS") else ["*"]
    )
    allow_headers: List[str] = Field(
        default_factory=lambda: os.getenv("CORS_ALLOW_HEADERS", "*").split(",")
        if os.getenv("CORS_ALLOW_HEADERS") else ["*"]
    )
    allow_credentials: bool = Field(
        default=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    )

settings = Settings()


# Domain models and schemas

class NoteBase(BaseModel):
    """Base schema for Note content."""
    title: str = Field(..., min_length=1, max_length=255, description="Title of the note")
    content: str = Field(..., min_length=1, description="Content/body of the note")

# PUBLIC_INTERFACE
class NoteCreate(NoteBase):
    """Schema for create note request."""


# PUBLIC_INTERFACE
class NoteUpdate(BaseModel):
    """Schema for update note request."""
    title: Optional[str] = Field(None, min_length=1, max_length=255, description="Updated title of the note")
    content: Optional[str] = Field(None, min_length=1, description="Updated content of the note")


# PUBLIC_INTERFACE
class Note(NoteBase):
    """Schema representing a stored Note."""
    id: int = Field(..., description="Unique identifier for the note")
    created_at: datetime = Field(..., description="Creation timestamp in UTC")
    updated_at: datetime = Field(..., description="Update timestamp in UTC")


# Repository protocol and implementations

class AbstractNotesRepository:
    """Abstract repository for notes."""

    # PUBLIC_INTERFACE
    async def list_notes(self, q: Optional[str] = None) -> List[Note]:
        """List all notes, optionally filtered by substring q in title or content."""
        raise NotImplementedError

    # PUBLIC_INTERFACE
    async def get_note(self, note_id: int) -> Optional[Note]:
        """Get a note by ID."""
        raise NotImplementedError

    # PUBLIC_INTERFACE
    async def create_note(self, data: NoteCreate) -> Note:
        """Create a new note from the provided data."""
        raise NotImplementedError

    # PUBLIC_INTERFACE
    async def update_note(self, note_id: int, data: NoteUpdate) -> Optional[Note]:
        """Update an existing note by ID with the provided data."""
        raise NotImplementedError

    # PUBLIC_INTERFACE
    async def delete_note(self, note_id: int) -> bool:
        """Delete a note by ID. Returns True if deleted, False if not found."""
        raise NotImplementedError


class InMemoryNotesRepository(AbstractNotesRepository):
    """In-memory repository that can optionally persist to a JSON file between requests."""
    def __init__(self, file_path: Optional[str] = None) -> None:
        self._notes: dict[int, Note] = {}
        self._next_id: int = 1
        self._file_path = file_path
        # Attempt to load existing data if file is configured
        if self._file_path:
            self._load_from_file()

    def _load_from_file(self) -> None:
        import json
        try:
            if self._file_path and os.path.exists(self._file_path):
                with open(self._file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    notes = {}
                    for item in content.get("notes", []):
                        # Convert ISO strings to datetime
                        item["created_at"] = datetime.fromisoformat(item["created_at"])
                        item["updated_at"] = datetime.fromisoformat(item["updated_at"])
                        note = Note(**item)
                        notes[note.id] = note
                    self._notes = notes
                    # Compute next id
                    self._next_id = max(notes.keys()) + 1 if notes else 1
        except Exception:
            # If anything goes wrong, we start fresh but do not crash the app
            self._notes = {}
            self._next_id = 1

    def _persist_to_file(self) -> None:
        import json
        if not self._file_path:
            return
        try:
            dir_name = os.path.dirname(self._file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            serializable = {
                "notes": [
                    {
                        "id": n.id,
                        "title": n.title,
                        "content": n.content,
                        "created_at": n.created_at.isoformat(),
                        "updated_at": n.updated_at.isoformat(),
                    }
                    for n in self._notes.values()
                ]
            }
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception:
            # Best-effort persistence; avoid raising to keep API responsive
            pass

    async def list_notes(self, q: Optional[str] = None) -> List[Note]:
        if not q:
            return sorted(self._notes.values(), key=lambda n: n.id)
        q_lower = q.lower()
        filtered = [
            n for n in self._notes.values()
            if q_lower in n.title.lower() or q_lower in n.content.lower()
        ]
        return sorted(filtered, key=lambda n: n.id)

    async def get_note(self, note_id: int) -> Optional[Note]:
        return self._notes.get(note_id)

    async def create_note(self, data: NoteCreate) -> Note:
        now = datetime.utcnow()
        note = Note(
            id=self._next_id,
            title=data.title,
            content=data.content,
            created_at=now,
            updated_at=now,
        )
        self._notes[note.id] = note
        self._next_id += 1
        self._persist_to_file()
        return note

    async def update_note(self, note_id: int, data: NoteUpdate) -> Optional[Note]:
        n = self._notes.get(note_id)
        if not n:
            return None
        updated = n.model_copy(update={
            "title": data.title if data.title is not None else n.title,
            "content": data.content if data.content is not None else n.content,
            "updated_at": datetime.utcnow(),
        })
        self._notes[note_id] = updated
        self._persist_to_file()
        return updated

    async def delete_note(self, note_id: int) -> bool:
        existed = note_id in self._notes
        if existed:
            del self._notes[note_id]
            self._persist_to_file()
        return existed


# Dependency injection

def _parse_file_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    # Accept formats: file:///abs/path or file://relative/path or file:/relative/path
    # We'll normalize to local path
    if url.startswith("file://"):
        path = url[len("file://"):]
        return path
    if url.startswith("file:"):
        path = url[len("file:"):]
        return path
    # Unsupported scheme -> ignore and return None (falls back to in-memory only)
    return None

_repo_instance: Optional[AbstractNotesRepository] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan to initialize repository (and potential resources)."""
    global _repo_instance
    file_path = _parse_file_url(settings.notes_db_url)
    _repo_instance = InMemoryNotesRepository(file_path=file_path)
    yield
    # No teardown needed for in-memory repo


def get_repository() -> AbstractNotesRepository:
    """FastAPI dependency for retrieving the notes repository instance."""
    if _repo_instance is None:
        # Should not happen under normal FastAPI lifecycle, but safe-guard:
        file_path = _parse_file_url(settings.notes_db_url)
        return InMemoryNotesRepository(file_path=file_path)
    return _repo_instance


# FastAPI app with OpenAPI metadata and tags
app = FastAPI(
    title=settings.app_name,
    description="A simple FastAPI backend that manages text notes with CRUD operations. "
                "Optionally persists to a local JSON file when NOTES_DB_URL=file://path is provided.",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "Health check and service info."},
        {"name": "Notes", "description": "CRUD operations for managing notes."}
    ],
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=settings.allow_methods,
    allow_headers=settings.allow_headers,
)


# Routes

@app.get(
    "/",
    tags=["Health"],
    summary="Health Check",
    description="Simple health check endpoint to verify the service is running.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {"application/json": {}}
        }
    },
)
def health_check():
    """Return a simple health check payload including environment information."""
    return {"message": "Healthy", "environment": settings.environment, "app": settings.app_name}


# PUBLIC_INTERFACE
@app.get(
    "/api/notes",
    response_model=List[Note],
    tags=["Notes"],
    summary="List notes",
    description="Retrieve all notes, optionally filtering by a query string contained in the title or content.",
)
async def list_notes(
    q: Optional[str] = Query(default=None, description="Optional filter to match notes by title/content substring."),
    repo: AbstractNotesRepository = Depends(get_repository),
):
    """List notes with optional simple substring search."""
    return await repo.list_notes(q=q)


# PUBLIC_INTERFACE
@app.get(
    "/api/notes/{note_id}",
    response_model=Note,
    tags=["Notes"],
    summary="Get note",
    description="Retrieve a single note by its ID.",
    responses={
        404: {"description": "Note not found."}
    }
)
async def get_note(
    note_id: int = Path(..., ge=1, description="ID of the note"),
    repo: AbstractNotesRepository = Depends(get_repository),
):
    """Fetch a single note by ID."""
    note = await repo.get_note(note_id)
    if not note:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    return note


# PUBLIC_INTERFACE
@app.post(
    "/api/notes",
    response_model=Note,
    status_code=status.HTTP_201_CREATED,
    tags=["Notes"],
    summary="Create note",
    description="Create a new note by providing title and content.",
)
async def create_note(
    payload: NoteCreate,
    repo: AbstractNotesRepository = Depends(get_repository),
):
    """Create a new note."""
    note = await repo.create_note(payload)
    return note


# PUBLIC_INTERFACE
@app.put(
    "/api/notes/{note_id}",
    response_model=Note,
    tags=["Notes"],
    summary="Update note",
    description="Update an existing note. Only provided fields will be updated.",
    responses={
        404: {"description": "Note not found."}
    }
)
async def update_note(
    note_id: int = Path(..., ge=1, description="ID of the note to update"),
    payload: NoteUpdate = None,
    repo: AbstractNotesRepository = Depends(get_repository),
):
    """Update a note by ID with provided fields."""
    if payload is None or (payload.title is None and payload.content is None):
        # Nothing to update
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided to update")
    updated = await repo.update_note(note_id, payload)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    return updated


# PUBLIC_INTERFACE
@app.delete(
    "/api/notes/{note_id}",
    tags=["Notes"],
    summary="Delete note",
    description="Delete a note by its ID.",
    responses={
        204: {"description": "Note deleted."},
        404: {"description": "Note not found."}
    }
)
async def delete_note(
    note_id: int = Path(..., ge=1, description="ID of the note to delete"),
    repo: AbstractNotesRepository = Depends(get_repository),
):
    """Delete a note by ID."""
    deleted = await repo.delete_note(note_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)
