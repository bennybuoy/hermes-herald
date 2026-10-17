"""Durable caller-owned identities for deliberate POST /v1/runs retries.

A committed reservation precedes transport. Leases serialize callers without
holding a SQLite lock over HTTP; abandoned attempts may only replay against an
advertised durable target within the original conservative retention horizon.
Tombstones are retained: forgetting an identity would make unsafe reuse new work.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from contextlib import closing
from http.client import HTTPException

from . import config as cfg, ledger

_REQUEST_FIELDS = {
    "profile", "message", "instructions", "delivery", "model", "provider",
    "reasoning_effort", "trace_id", "parent_edge_id", "parent_hop", "max_hops",
}

LEASE_SECONDS = 120
RETENTION_MARGIN_SECONDS = 60


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class IdentityError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class DispatchIdentity:
    def __init__(self, args):
        self.key = args.get("idempotency_key") if "idempotency_key" in args else uuid.uuid4().hex
        if (not isinstance(self.key, str) or not 1 <= len(self.key) <= 255
                or any(not 33 <= ord(ch) <= 126 for ch in self.key)):
            raise IdentityError("invalid_idempotency_key", "idempotency_key must be 1-255 visible ASCII characters; it is never trimmed or repaired.")
        self.args = {k: v for k, v in args.items() if k in _REQUEST_FIELDS}
        self.token = uuid.uuid4().hex
        self.row = None
        self.claimed = False
        self.accepted = False
        self.accepted_response = None
        self.server_conflict = False
        self.capability_warning = ""

    def prepare(self, pcfg, routing):
        request = _encode(self.args)
        owner = _encode({
            "home": str(cfg.get_hermes_home().resolve()),
            "origin": cfg.get_active_profile_name(),
            "session": routing.get("parent_session_id") or routing.get("session_id") or routing.get("session_key") or "",
            "session_key": routing.get("session_key") or "",
        })
        route = _encode({
            "url": pcfg["url"], "model": pcfg.get("model"),
            # The target scopes reservations to profile + bearer key. A rotated
            # credential must not silently enter a new server namespace.
            "credential_sha256": hashlib.sha256(pcfg["api_key"].encode()).hexdigest(),
        })
        now = time.time()
        with closing(ledger._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM dispatch_identities WHERE idempotency_key=?", (self.key,)).fetchone()
            if row is None:
                edge_id = uuid.uuid4().hex
                trace_id = str(self.args.get("trace_id") or "").strip() or uuid.uuid4().hex
                db.execute(
                    "INSERT INTO dispatch_identities (idempotency_key, request_json, owner_json, route_json, edge_id, trace_id, created_at, lease_until, attempt_token) VALUES (?,?,?,?,?,?,?,?,?)",
                    (self.key, request, owner, route, edge_id, trace_id, now, now + LEASE_SECONDS, self.token),
                )
            else:
                self.row = dict(row)
                if (row["request_json"], row["owner_json"], row["route_json"]) != (request, owner, route):
                    raise IdentityError("idempotency_conflict", "This key belongs to different arguments, routing, or origin/session ownership. Do not change the request when retrying.")
                if row["response_json"]:
                    response = json.loads(row["response_json"])
                    response["idempotency_replayed"] = True
                    if self.args.get("delivery", "callback") == "callback":
                        response["callback"] = "Original acceptance returned; monitoring is process-bound and no listener is restarted. Use check_dispatch if needed."
                    return response
                if row["blocked_code"]:
                    raise IdentityError(row["blocked_code"], "Target returned HTTP 409 for this identity. Reconcile existing work; do not rotate the key to bypass the conflict.")
                if row["lease_until"] > now:
                    raise IdentityError("idempotency_in_progress", "This identity already has an in-flight caller. Wait and reuse the same key and original arguments.")
                if row["body_json"] and row["replay_until"] <= now:
                    raise IdentityError("idempotency_unsafe_replay", "The original target did not advertise durable replay, or its safe retention horizon has expired. Reconcile remotely; no POST was sent.")
                db.execute("UPDATE dispatch_identities SET lease_until=?, attempt_token=? WHERE idempotency_key=?", (now + LEASE_SECONDS, self.token, self.key))
            self.row = dict(db.execute("SELECT * FROM dispatch_identities WHERE idempotency_key=?", (self.key,)).fetchone())
        self.claimed = True
        return None

    def submit(self, pcfg, body, post, get):
        encoded = _encode(body)
        old_body = self.row["body_json"]
        if old_body and old_body != encoded:
            raise IdentityError("idempotency_conflict", "Resolved target routing or request body changed. The original body cannot be altered on replay.")
        capability = {}
        try:
            payload = get(f"{pcfg['url']}/v1/capabilities", pcfg["api_key"], timeout=5.0)
            capability = payload.get("features", {}).get("runs_idempotency", {})
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError, HTTPException):
            # Initial dispatch stays compatible with older targets. Unknown
            # support can never authorize replay of an ambiguous submission.
            capability = {}
        retention = capability.get("retention_seconds") if isinstance(capability, dict) else None
        durable = (isinstance(capability, dict) and capability.get("supported") is True
                   and capability.get("durable") is True
                   and type(retention) in (int, float) and math.isfinite(retention)
                   and retention > RETENTION_MARGIN_SECONDS)
        now = time.time()
        horizon = self.row["created_at"] + retention - RETENTION_MARGIN_SECONDS if durable else 0
        if old_body and (not durable or min(horizon, self.row["replay_until"]) <= now):
            raise IdentityError("idempotency_unsafe_replay", "Target durable replay capability is unavailable or the original retention horizon expired. Reconcile remotely; no POST was sent.")
        if not durable:
            self.capability_warning = "Target did not advertise durable run idempotency. Initial submission is allowed, but an ambiguous response cannot safely be replayed."
        with closing(ledger._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            changed = db.execute(
                "UPDATE dispatch_identities SET body_json=?, replay_until=? WHERE idempotency_key=? AND attempt_token=? AND lease_until>? AND response_json IS NULL",
                (encoded, min(horizon, self.row["replay_until"]) if old_body else horizon, self.key, self.token, now),
            ).rowcount
            if changed != 1:
                raise IdentityError("idempotency_in_progress", "The reservation lease no longer belongs to this caller. Reuse the key later; no POST was sent.")
        try:
            result = post(f"{pcfg['url']}/v1/runs", pcfg["api_key"], json.loads(old_body or encoded), timeout=30.0, idempotency_key=self.key)
        except RuntimeError as exc:
            self.server_conflict = getattr(exc, "status_code", None) == 409
            raise
        run_id = result.get("run_id") if isinstance(result, dict) else None
        if not isinstance(run_id, str) or not run_id:
            raise IdentityError("idempotency_ambiguous_response", "Target response did not contain a valid run_id. Acceptance is unknown; reuse the original key and arguments only if safe replay is available.")
        # Commit the original handle before listener/state side effects. A crash
        # in that gap returns this handle on retry, never duplicates delivery.
        response = {
            "run_id": run_id, "profile": self.args["profile"],
            "status": "dispatched_unmonitored", "delivery": self.args.get("delivery", "callback"),
            "edge_id": self.row["edge_id"], "trace_id": self.row["trace_id"],
            "warning": "Recovered accepted run. Local audit/cache/monitoring may be incomplete; use check_dispatch. No listener is restarted on retry.",
        }
        self.accepted = True
        self.accepted_response = response
        self._store_response(response)
        return result

    def _store_response(self, response):
        with closing(ledger._connect()) as db, db:
            changed = db.execute("UPDATE dispatch_identities SET response_json=? WHERE idempotency_key=? AND attempt_token=?", (_encode(response), self.key, self.token)).rowcount
            if changed != 1:
                self.claimed = False
                raise IdentityError("idempotency_in_progress", "Another caller now owns this reservation. The known run handle is retained, but no local listener/audit/cache side effects were started by this caller.")

    def finish(self, response):
        response["idempotency_key"] = self.key
        response["retry_instruction"] = "To retry this work, reuse this exact idempotency_key with all original arguments in the owning origin/session. Omitting the key means NEW work."
        if self.capability_warning:
            response["warning"] = " ".join(filter(None, (response.get("warning"), self.capability_warning)))
        if not self.claimed:
            return response
        if self.accepted:
            self._store_response(response)
        else:
            if self.server_conflict:
                response["error_code"] = "idempotency_server_conflict"
                response["error"] = "Target HTTP 409 idempotency conflict. Reconcile existing work; do not rotate the key to bypass it."
            with closing(ledger._connect()) as db, db:
                db.execute("UPDATE dispatch_identities SET lease_until=0, blocked_code=? WHERE idempotency_key=? AND attempt_token=?", ("idempotency_server_conflict" if self.server_conflict else "", self.key, self.token))
        return response
