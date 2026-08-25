"""Where a check goes after it has been run, so a person can open it again.

A tool that shows you a result once and then loses it is a demo, not something
anyone can work with. This module is the store behind the history page.

**Three things are deliberate.**

*It fails fast and it says so.* The store is MongoDB Atlas, which is on the
other side of a network. pymongo's default server-selection timeout is thirty
seconds, which in front of a panel is indistinguishable from a hung server, so
it is set to three. Every failure returns `available: false` with a sentence
saying what went wrong rather than a 500, because the history page working and
saying "the store is unreachable" is worth more than the history page not
loading.

*It never becomes a silent in-memory store.* An implementation that keeps
checks in a dict and loses them on restart would make every screen look right
and be a lie. When no connection string is configured the store reports itself
unavailable and the interface falls back to the browser's own storage, which is
real persistence and is labelled as what it is.

*The list is not the results.* `HistoryEntry` carries a preview and four counts.
Fifty saved checks are fifty documents and fifty span arrays, and a list view
has no use for any of it; the full result is fetched only when someone opens
one.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Tuple

from src.common.schema import (
    AnalysisResult,
    HistoryDetail,
    HistoryEntry,
    TaskType,
)

# Long enough to cross a network, short enough that a dead one is obvious while
# someone is still looking at the screen.
SERVER_SELECTION_TIMEOUT_MS = 3000
PREVIEW_CHARS = 240
DEFAULT_LIMIT = 25
MAX_LIMIT = 200
COLLECTION = "checks"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """The first line-ish of an answer, for the list view.

    Whitespace is collapsed first: an answer that begins with a newline would
    otherwise produce a preview that renders as an empty row.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def entry_from_result(result: AnalysisResult, entry_id: str, created_at: str) -> HistoryEntry:
    """The list row for one result. Pure, so it can be tested without a server."""
    flagged = sum(1 for s in result.spans if s.conformal_decision == "flag")
    unsure = sum(1 for s in result.spans if s.conformal_decision == "abstain")
    return HistoryEntry(
        id=entry_id,
        created_at=created_at,
        task_type=result.task_type,
        question=" ".join(result.question.split()),
        answer_preview=preview(result.answer),
        n_spans=len(result.spans),
        n_flagged=flagged,
        n_unsure=unsure,
        guarantee_applies=result.guarantee_applies,
        alpha=result.alpha,
        model_version=result.model_version,
    )


def build_filter(
    q: Optional[str] = None,
    task: Optional[str] = None,
    only_flagged: bool = False,
) -> Dict[str, Any]:
    """The Mongo query for a set of filters.

    Split out and pure so the filter logic is tested without a database. The
    search term is escaped: a person typing `c++` or `(` into a search box must
    not be able to hand a regex to the query planner.
    """
    clauses: List[Dict[str, Any]] = []
    if q:
        pattern = re.escape(q.strip())
        if pattern:
            clauses.append(
                {
                    "$or": [
                        {"question": {"$regex": pattern, "$options": "i"}},
                        {"answer_preview": {"$regex": pattern, "$options": "i"}},
                    ]
                }
            )
    if task and task != "all":
        # Rejected rather than passed through: an unknown task would silently
        # return nothing, which reads as "you have no history".
        if task not in {t.value for t in TaskType}:
            raise ValueError(f"unknown task type {task!r}")
        clauses.append({"task_type": task})
    if only_flagged:
        clauses.append({"n_flagged": {"$gt": 0}})

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def clamp_limit(limit: Optional[int]) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, int(limit)))


class HistoryStore(Protocol):
    """What the API needs from a store, and nothing more."""

    backend: str

    def available(self) -> bool: ...

    def note(self) -> str: ...

    def save(self, result: AnalysisResult) -> HistoryEntry: ...

    def list(
        self,
        q: Optional[str] = None,
        task: Optional[str] = None,
        only_flagged: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Tuple[List[HistoryEntry], int]: ...

    def get(self, entry_id: str) -> Optional[HistoryDetail]: ...

    def delete(self, entry_id: str) -> bool: ...

    def clear(self) -> int: ...

    def export(self) -> List[Dict[str, Any]]: ...


class UnavailableStore:
    """The store when there is none.

    It is not an error. No connection string configured is the normal state of a
    fresh checkout, and the interface has a real answer for it -- keep the
    history in the browser instead. This exists so that path is explicit rather
    than being an exception someone has to catch.
    """

    backend = "none"

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def available(self) -> bool:
        return False

    def note(self) -> str:
        return self._reason

    def save(self, result: AnalysisResult) -> HistoryEntry:
        raise RuntimeError(self._reason)

    def list(self, *_args, **_kwargs) -> Tuple[List[HistoryEntry], int]:
        return [], 0

    def get(self, entry_id: str) -> Optional[HistoryDetail]:
        return None

    def delete(self, entry_id: str) -> bool:
        return False

    def clear(self) -> int:
        return 0

    def export(self) -> List[Dict[str, Any]]:
        return []


class MongoHistoryStore:
    """Checks in a MongoDB collection.

    One document per check: the list fields at the top level so they can be
    filtered and sorted, and the whole `AnalysisResult` nested under `result`.
    That nesting is the reason this is a document store rather than a table --
    a result is a tree with a span array in it, and flattening it into columns
    would buy nothing.
    """

    backend = "mongodb"

    def __init__(self, uri: str, database: str = "trustrag") -> None:
        from pymongo import DESCENDING, MongoClient

        self._client = MongoClient(
            uri,
            serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
            connectTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
            appname="trustrag",
        )
        self._col = self._client[database][COLLECTION]
        self._descending = DESCENDING
        self._reason = ""

    # -- connection ---------------------------------------------------------

    def available(self) -> bool:
        try:
            self._client.admin.command("ping")
        except Exception as exc:  # noqa: BLE001
            self._reason = f"the history store did not answer: {exc.__class__.__name__}"
            return False
        self._reason = ""
        return True

    def note(self) -> str:
        return self._reason

    def ensure_indexes(self) -> None:
        """Newest-first is the only ordering this collection is ever read in.

        Never fatal. An index that could not be created costs speed on a
        collection that will hold hundreds of documents, and refusing to start
        over it would cost the demo.
        """
        try:
            self._col.create_index(
                [("created_at", self._descending), ("_id", self._descending)]
            )
        except Exception:  # noqa: BLE001
            pass

    # -- writes -------------------------------------------------------------

    def save(self, result: AnalysisResult) -> HistoryEntry:
        created_at = _now_iso()
        document = {
            "created_at": created_at,
            "task_type": result.task_type.value
            if hasattr(result.task_type, "value")
            else str(result.task_type),
            "question": " ".join(result.question.split()),
            "answer_preview": preview(result.answer),
            "n_spans": len(result.spans),
            "n_flagged": sum(1 for s in result.spans if s.conformal_decision == "flag"),
            "n_unsure": sum(
                1 for s in result.spans if s.conformal_decision == "abstain"
            ),
            "guarantee_applies": result.guarantee_applies,
            "alpha": result.alpha,
            "model_version": result.model_version,
            "result": result.model_dump(mode="json"),
        }
        inserted = self._col.insert_one(document)
        return entry_from_result(result, str(inserted.inserted_id), created_at)

    def delete(self, entry_id: str) -> bool:
        key = self._key(entry_id)
        if key is None:
            return False
        return self._col.delete_one({"_id": key}).deleted_count == 1

    def clear(self) -> int:
        return self._col.delete_many({}).deleted_count

    # -- reads --------------------------------------------------------------

    def list(
        self,
        q: Optional[str] = None,
        task: Optional[str] = None,
        only_flagged: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Tuple[List[HistoryEntry], int]:
        query = build_filter(q, task, only_flagged)
        total = self._col.count_documents(query)
        cursor = (
            self._col.find(query, {"result": 0})
            # `_id` breaks the tie. Two checks saved inside the same second
            # share a timestamp, and without a second key their order is
            # whatever the engine felt like -- which shows up as a list that
            # reshuffles itself between page loads. ObjectIds are monotonic, so
            # newest-id is newest.
            .sort([("created_at", self._descending), ("_id", self._descending)])
            .skip(max(0, int(offset)))
            .limit(clamp_limit(limit))
        )
        return [self._entry(doc) for doc in cursor], total

    def get(self, entry_id: str) -> Optional[HistoryDetail]:
        key = self._key(entry_id)
        if key is None:
            return None
        doc = self._col.find_one({"_id": key})
        if doc is None:
            return None
        return HistoryDetail(
            entry=self._entry(doc),
            result=AnalysisResult.model_validate(doc["result"]),
        )

    def export(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for doc in self._col.find({}).sort(
            [("created_at", self._descending), ("_id", self._descending)]
        ):
            doc["id"] = str(doc.pop("_id"))
            out.append(doc)
        return out

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _key(entry_id: str) -> Optional[Any]:
        """An id from the URL, as an ObjectId -- or nothing.

        A malformed id is a 404, not a 500. The route hands whatever was in the
        path straight to this.
        """
        from bson import ObjectId
        from bson.errors import InvalidId

        try:
            return ObjectId(entry_id)
        except (InvalidId, TypeError):
            return None

    @staticmethod
    def _entry(doc: Dict[str, Any]) -> HistoryEntry:
        return HistoryEntry(
            id=str(doc["_id"]),
            created_at=doc.get("created_at", ""),
            task_type=doc.get("task_type", "other"),
            question=doc.get("question", ""),
            answer_preview=doc.get("answer_preview", ""),
            n_spans=doc.get("n_spans", 0),
            n_flagged=doc.get("n_flagged", 0),
            n_unsure=doc.get("n_unsure", 0),
            guarantee_applies=doc.get("guarantee_applies", True),
            alpha=doc.get("alpha"),
            model_version=doc.get("model_version", ""),
        )


_store: Optional[HistoryStore] = None


def build_store() -> HistoryStore:
    """The store this process will use, from the environment.

    Never raises. Every way this can fail -- no connection string, pymongo not
    installed, a bad URI, an unreachable cluster -- ends as an `UnavailableStore`
    carrying a sentence the interface can show. The API is not the right place
    to find out that someone has not signed up for Atlas yet.
    """
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        return UnavailableStore(
            "No MONGODB_URI is set, so this server is not keeping any history. "
            "Checks are being kept in this browser instead."
        )
    try:
        store = MongoHistoryStore(uri, os.environ.get("MONGODB_DB", "trustrag"))
    except ImportError:
        return UnavailableStore(
            "pymongo is not installed. Run `pip install -r requirements.txt`."
        )
    except Exception as exc:  # noqa: BLE001
        return UnavailableStore(f"the history store could not be opened: {exc}")
    store.ensure_indexes()
    return store


def get_store() -> HistoryStore:
    global _store
    if _store is None:
        _store = build_store()
    return _store


def set_store(store: Optional[HistoryStore]) -> None:
    """Swap the store. For tests, and for a reconnect after the URI changes."""
    global _store
    _store = store
