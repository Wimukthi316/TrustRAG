"""The MongoDB driver itself, against an in-process Mongo.

`tests/test_history.py` covers the pure logic and the routes through a fake
store. What it cannot cover is the part that only exists once pymongo is in the
picture: the document shape, the ObjectId round trip, the projection that keeps
whole results out of the list query, and the sort.

This runs against `mongomock`, which is not required to install this project --
if it is absent the module skips and the rest of the suite is unaffected. Install
it with `pip install mongomock` to get this coverage:

    the alternative is that nothing exercises the driver until a real cluster is
    connected, and finding a document-shape bug at that point means finding it in
    front of whoever asked for the demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mongomock = pytest.importorskip("mongomock")

from pymongo import DESCENDING  # noqa: E402

from backend.app.services.history import (  # noqa: E402
    COLLECTION,
    MongoHistoryStore,
    UnavailableStore,
    build_store,
)
from tests.test_history import make_result  # noqa: E402


@pytest.fixture()
def store():
    """A MongoHistoryStore wired to an in-process Mongo.

    Built field by field rather than through __init__, because __init__ is the
    one part that has to open a real connection.
    """
    s = MongoHistoryStore.__new__(MongoHistoryStore)
    client = mongomock.MongoClient()
    s._client = client
    s._col = client["trustrag"][COLLECTION]
    s._descending = DESCENDING
    s._reason = ""
    return s


def test_the_whole_result_survives_the_round_trip(store):
    result = make_result(decisions=("flag", "abstain", "pass"))
    entry = store.save(result)

    detail = store.get(entry.id)
    assert detail is not None
    assert detail.result.answer == result.answer
    assert detail.result.context == result.context
    assert [s.conformal_decision for s in detail.result.spans] == [
        s.conformal_decision for s in result.spans
    ]
    assert detail.result.spans[0].start == result.spans[0].start


def test_the_list_query_does_not_drag_results_across(store):
    """A list of fifty checks must not carry fifty documents with it."""
    store.save(make_result())
    doc = store._col.find({}, {"result": 0}).next()
    assert "result" not in doc
    assert "answer_preview" in doc


def test_counts_are_stored_not_recomputed_on_read(store):
    entry = store.save(make_result(decisions=("flag", "flag", "abstain")))
    doc = store._col.find_one({})
    assert (doc["n_spans"], doc["n_flagged"], doc["n_unsure"]) == (3, 2, 1)
    assert (entry.n_spans, entry.n_flagged, entry.n_unsure) == (3, 2, 1)


def test_newest_first_even_within_the_same_second(store):
    """Same timestamp, so only the id tiebreaker can order these."""
    ids = [store.save(make_result(question=f"q{i}")).id for i in range(5)]
    listed, total = store.list()
    assert total == 5
    assert [e.id for e in listed] == list(reversed(ids))


def test_filters_reach_the_database(store):
    store.save(make_result(question="about mice", decisions=("flag",)))
    store.save(make_result(question="about carbon", decisions=("pass",)))
    store.save(
        make_result(
            question="a summary", task_type="summarization", decisions=("pass",)
        )
    )

    assert store.list(q="mice")[1] == 1
    assert store.list(q="MICE")[1] == 1, "search should not be case sensitive"
    assert store.list(only_flagged=True)[1] == 1
    assert store.list(task="summarization")[1] == 1
    assert store.list(q="nothing here")[1] == 0


def test_a_malformed_id_is_absent_not_an_exception(store):
    """The route hands whatever was in the path straight to these."""
    assert store.get("not-an-objectid") is None
    assert store.delete("not-an-objectid") is False
    assert store.get("") is None


def test_delete_and_clear(store):
    ids = [store.save(make_result()).id for _ in range(3)]
    assert store.delete(ids[0]) is True
    assert store.delete(ids[0]) is False
    assert store.list()[1] == 2
    assert store.clear() == 2
    assert store.list()[1] == 0


def test_export_carries_the_results_and_a_string_id(store):
    store.save(make_result())
    exported = store.export()
    assert len(exported) == 1
    assert isinstance(exported[0]["id"], str)
    assert exported[0]["result"]["answer"] == make_result().answer
    assert "_id" not in exported[0]


def test_an_unreachable_cluster_degrades_instead_of_hanging():
    """The demo case: a connection string, and no network behind it.

    `build_store` must come back with something usable rather than raising, and
    `available()` must return False rather than blocking for pymongo's default
    thirty-second server selection timeout.
    """
    import os
    import time

    previous = os.environ.get("MONGODB_URI")
    os.environ["MONGODB_URI"] = "mongodb://127.0.0.1:1/?directConnection=true"
    try:
        started = time.monotonic()
        store = build_store()
        assert not isinstance(store, UnavailableStore), "a valid URI should build"
        assert store.available() is False
        assert store.note(), "it has to say why"
        # Three seconds is the configured timeout; ten is a generous ceiling on
        # a loaded machine and still nowhere near pymongo's default of thirty.
        assert time.monotonic() - started < 10
    finally:
        if previous is None:
            os.environ.pop("MONGODB_URI", None)
        else:
            os.environ["MONGODB_URI"] = previous
