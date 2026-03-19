#!/usr/bin/env python3
"""Lightweight, hackable IMAP/SMTP client for the CTO-Agent."""

from __future__ import annotations

import argparse
import email
import imaplib
import json
import os
import smtplib
import sqlite3
import ssl
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import default as default_policy
from email.utils import getaddresses, make_msgid, parsedate_to_datetime
from pathlib import Path
from typing import Iterable


DEFAULT_DB_PATH = Path("runtime/cto_agent.db")
DEFAULT_IMAP_HOST = "imap.one.com"
DEFAULT_IMAP_PORT = 993
DEFAULT_SMTP_HOST = "send.one.com"
DEFAULT_SMTP_PORT = 465
DEFAULT_SENT_FOLDER = "sent"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_mime_header(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def normalize_addresses(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for _name, address in getaddresses(list(values)):
        address = (address or "").strip().lower()
        if address and address not in out:
            out.append(address)
    return out


def extract_body_parts(message: email.message.Message) -> tuple[str, str]:
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if message.is_multipart():
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get("Content-Disposition", "").lower().startswith("attachment"):
                continue
            content_type = part.get_content_type()
            try:
                payload = part.get_content()
            except Exception:
                raw = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                payload = raw.decode(charset, errors="replace")
            text = str(payload or "").strip()
            if not text:
                continue
            if content_type == "text/plain":
                plain_parts.append(text)
            elif content_type == "text/html":
                html_parts.append(text)
    else:
        try:
            payload = message.get_content()
        except Exception:
            raw = message.get_payload(decode=True) or b""
            charset = message.get_content_charset() or "utf-8"
            payload = raw.decode(charset, errors="replace")
        text = str(payload or "").strip()
        if message.get_content_type() == "text/html":
            html_parts.append(text)
        else:
            plain_parts.append(text)

    body_text = "\n\n".join(part for part in plain_parts if part).strip()
    body_html = "\n\n".join(part for part in html_parts if part).strip()
    return body_text[:200000], body_html[:200000]


def preview_text(body_text: str, subject: str) -> str:
    source = body_text or subject or ""
    collapsed = " ".join(source.split())
    return collapsed[:280]


def parse_message_datetime(message: email.message.Message) -> tuple[str, int]:
    date_header = (message.get("Date") or "").strip()
    if not date_header:
        return "", 0
    try:
        dt = parsedate_to_datetime(date_header)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(), int(dt.timestamp())
    except Exception:
        return date_header, 0


@dataclass
class StoredMessage:
    record_id: str
    account_id: str
    folder: str
    direction: str
    uid: str
    message_id_header: str
    subject: str
    from_name: str
    from_email: str
    to_emails_json: str
    cc_emails_json: str
    received_at_iso: str
    received_at_ts: int
    seen: int
    preview: str
    body_text: str
    body_html: str
    raw_size: int
    synced_at: str
    metadata_json: str


def stored_message_from_imap(
    account_id: str,
    folder: str,
    uid: str,
    raw_bytes: bytes,
    flags: list[str],
) -> StoredMessage:
    parsed = BytesParser(policy=default_policy).parsebytes(raw_bytes)
    message_id_header = decode_mime_header(parsed.get("Message-ID")) or f"<imap-{account_id}-{folder}-{uid}>"
    subject = decode_mime_header(parsed.get("Subject")) or "(ohne Betreff)"
    from_pairs = getaddresses([parsed.get("From") or ""])
    from_name = ""
    from_email = ""
    if from_pairs:
        from_name = decode_mime_header(from_pairs[0][0] or "")
        from_email = (from_pairs[0][1] or "").strip().lower()

    body_text, body_html = extract_body_parts(parsed)
    received_at_iso, received_at_ts = parse_message_datetime(parsed)
    to_emails = normalize_addresses([parsed.get("To") or ""])
    cc_emails = normalize_addresses([parsed.get("Cc") or ""])
    seen = 1 if "\\Seen" in flags else 0
    synced_at = now_iso()
    metadata = {
        "message_id_header": message_id_header,
        "references": decode_mime_header(parsed.get("References")),
        "in_reply_to": decode_mime_header(parsed.get("In-Reply-To")),
        "flags": flags,
    }
    return StoredMessage(
        record_id=f"{account_id}::{folder}::{uid}",
        account_id=account_id,
        folder=folder,
        direction="inbound",
        uid=uid,
        message_id_header=message_id_header,
        subject=subject,
        from_name=from_name,
        from_email=from_email,
        to_emails_json=json.dumps(to_emails, ensure_ascii=True),
        cc_emails_json=json.dumps(cc_emails, ensure_ascii=True),
        received_at_iso=received_at_iso,
        received_at_ts=received_at_ts,
        seen=seen,
        preview=preview_text(body_text, subject),
        body_text=body_text,
        body_html=body_html,
        raw_size=len(raw_bytes),
        synced_at=synced_at,
        metadata_json=json.dumps(metadata, ensure_ascii=True),
    )


def stored_message_from_outbound(
    account_id: str,
    sender_email: str,
    to_emails: list[str],
    cc_emails: list[str],
    subject: str,
    body_text: str,
    message_id_header: str,
) -> StoredMessage:
    sent_at = now_iso()
    sent_ts = int(datetime.now(timezone.utc).timestamp())
    metadata = {"message_id_header": message_id_header}
    return StoredMessage(
        record_id=f"{account_id}::outbound::{message_id_header}",
        account_id=account_id,
        folder=DEFAULT_SENT_FOLDER,
        direction="outbound",
        uid="",
        message_id_header=message_id_header,
        subject=subject or "(ohne Betreff)",
        from_name="",
        from_email=sender_email.strip().lower(),
        to_emails_json=json.dumps(to_emails, ensure_ascii=True),
        cc_emails_json=json.dumps(cc_emails, ensure_ascii=True),
        received_at_iso=sent_at,
        received_at_ts=sent_ts,
        seen=1,
        preview=preview_text(body_text, subject),
        body_text=body_text[:200000],
        body_html="",
        raw_size=len(body_text.encode("utf-8", errors="replace")),
        synced_at=sent_at,
        metadata_json=json.dumps(metadata, ensure_ascii=True),
    )


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA busy_timeout=5000;

        CREATE TABLE IF NOT EXISTS mail_accounts (
            id TEXT PRIMARY KEY,
            email_address TEXT NOT NULL,
            imap_host TEXT NOT NULL,
            imap_port INTEGER NOT NULL,
            smtp_host TEXT NOT NULL,
            smtp_port INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_imap_ok_at TEXT,
            last_smtp_ok_at TEXT
        );

        CREATE TABLE IF NOT EXISTS mail_messages (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            folder TEXT NOT NULL,
            direction TEXT NOT NULL,
            uid TEXT NOT NULL,
            message_id_header TEXT NOT NULL,
            subject TEXT NOT NULL,
            from_name TEXT NOT NULL,
            from_email TEXT NOT NULL,
            to_emails_json TEXT NOT NULL,
            cc_emails_json TEXT NOT NULL,
            received_at_iso TEXT NOT NULL,
            received_at_ts INTEGER NOT NULL,
            seen INTEGER NOT NULL,
            preview TEXT NOT NULL,
            body_text TEXT NOT NULL,
            body_html TEXT NOT NULL,
            raw_size INTEGER NOT NULL,
            synced_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_mail_messages_account_folder_ts
            ON mail_messages(account_id, folder, received_at_ts DESC);

        CREATE TABLE IF NOT EXISTS mail_sync_runs (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            folder TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            ok INTEGER NOT NULL,
            fetched_count INTEGER NOT NULL,
            stored_count INTEGER NOT NULL,
            error_text TEXT NOT NULL
        );
        """
    )
    conn.commit()


def upsert_account(
    conn: sqlite3.Connection,
    *,
    email_address: str,
    imap_host: str,
    imap_port: int,
    smtp_host: str,
    smtp_port: int,
    mark_imap_ok: bool = False,
    mark_smtp_ok: bool = False,
) -> str:
    account_id = email_address.strip().lower()
    now = now_iso()
    conn.execute(
        """
        INSERT INTO mail_accounts (
            id, email_address, imap_host, imap_port, smtp_host, smtp_port,
            created_at, updated_at, last_imap_ok_at, last_smtp_ok_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            email_address=excluded.email_address,
            imap_host=excluded.imap_host,
            imap_port=excluded.imap_port,
            smtp_host=excluded.smtp_host,
            smtp_port=excluded.smtp_port,
            updated_at=excluded.updated_at,
            last_imap_ok_at=COALESCE(excluded.last_imap_ok_at, mail_accounts.last_imap_ok_at),
            last_smtp_ok_at=COALESCE(excluded.last_smtp_ok_at, mail_accounts.last_smtp_ok_at)
        """,
        (
            account_id,
            account_id,
            imap_host,
            imap_port,
            smtp_host,
            smtp_port,
            now,
            now,
            now if mark_imap_ok else None,
            now if mark_smtp_ok else None,
        ),
    )
    conn.commit()
    return account_id


def upsert_message(conn: sqlite3.Connection, message: StoredMessage) -> None:
    conn.execute(
        """
        INSERT INTO mail_messages (
            id, account_id, folder, direction, uid, message_id_header,
            subject, from_name, from_email, to_emails_json, cc_emails_json,
            received_at_iso, received_at_ts, seen, preview, body_text, body_html,
            raw_size, synced_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            account_id=excluded.account_id,
            folder=excluded.folder,
            direction=excluded.direction,
            uid=excluded.uid,
            message_id_header=excluded.message_id_header,
            subject=excluded.subject,
            from_name=excluded.from_name,
            from_email=excluded.from_email,
            to_emails_json=excluded.to_emails_json,
            cc_emails_json=excluded.cc_emails_json,
            received_at_iso=excluded.received_at_iso,
            received_at_ts=excluded.received_at_ts,
            seen=excluded.seen,
            preview=excluded.preview,
            body_text=excluded.body_text,
            body_html=excluded.body_html,
            raw_size=excluded.raw_size,
            synced_at=excluded.synced_at,
            metadata_json=excluded.metadata_json
        """,
        (
            message.record_id,
            message.account_id,
            message.folder,
            message.direction,
            message.uid,
            message.message_id_header,
            message.subject,
            message.from_name,
            message.from_email,
            message.to_emails_json,
            message.cc_emails_json,
            message.received_at_iso,
            message.received_at_ts,
            message.seen,
            message.preview,
            message.body_text,
            message.body_html,
            message.raw_size,
            message.synced_at,
            message.metadata_json,
        ),
    )


def record_sync_run(
    conn: sqlite3.Connection,
    *,
    account_id: str,
    folder: str,
    started_at: str,
    finished_at: str,
    ok: bool,
    fetched_count: int,
    stored_count: int,
    error_text: str,
) -> None:
    conn.execute(
        """
        INSERT INTO mail_sync_runs (
            id, account_id, folder, started_at, finished_at, ok,
            fetched_count, stored_count, error_text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            account_id,
            folder,
            started_at,
            finished_at,
            1 if ok else 0,
            fetched_count,
            stored_count,
            error_text,
        ),
    )
    conn.commit()


def parse_fetch_flags(fetch_header: bytes) -> list[str]:
    text = fetch_header.decode("utf-8", errors="replace")
    if "FLAGS (" not in text:
        return []
    tail = text.split("FLAGS (", 1)[1]
    raw_flags = tail.split(")", 1)[0]
    return [flag for flag in raw_flags.split() if flag]


def require_password(args: argparse.Namespace) -> str:
    if args.password:
        return args.password
    env_name = args.password_env or "CTO_EMAIL_PASSWORD"
    value = os.environ.get(env_name, "")
    if value:
        return value
    raise SystemExit(f"Missing password. Set --password or export {env_name}.")


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    ensure_schema(conn)
    return conn


def imap_sync(args: argparse.Namespace) -> dict:
    password = require_password(args)
    db_path = Path(args.db)
    conn = connect_db(db_path)
    started_at = now_iso()
    account_id = upsert_account(
        conn,
        email_address=args.email,
        imap_host=args.imap_host,
        imap_port=args.imap_port,
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
    )

    fetched_count = 0
    stored_count = 0
    folder = args.folder

    try:
        with imaplib.IMAP4_SSL(args.imap_host, args.imap_port, ssl_context=ssl.create_default_context()) as imap:
            imap.login(args.email, password)
            status, _select_data = imap.select(folder, readonly=not args.mark_seen)
            if status != "OK":
                raise RuntimeError(f"IMAP select failed for folder {folder}")

            status, search_data = imap.uid("search", None, "ALL")
            if status != "OK":
                raise RuntimeError("IMAP UID search failed")
            uids = [item.decode("ascii", errors="ignore") for item in (search_data[0] or b"").split() if item]
            selected = list(reversed(uids[-args.limit :]))
            fetched_count = len(selected)

            for uid in selected:
                status, fetch_data = imap.uid("fetch", uid, "(RFC822 FLAGS)")
                if status != "OK" or not fetch_data:
                    continue
                raw_bytes = b""
                flags: list[str] = []
                for item in fetch_data:
                    if not isinstance(item, tuple):
                        continue
                    header, payload = item
                    if isinstance(payload, bytes):
                        raw_bytes += payload
                    if isinstance(header, bytes):
                        flags = parse_fetch_flags(header)
                if not raw_bytes:
                    continue
                message = stored_message_from_imap(account_id, folder, uid, raw_bytes, flags)
                upsert_message(conn, message)
                stored_count += 1

            conn.commit()
            upsert_account(
                conn,
                email_address=args.email,
                imap_host=args.imap_host,
                imap_port=args.imap_port,
                smtp_host=args.smtp_host,
                smtp_port=args.smtp_port,
                mark_imap_ok=True,
            )
    except Exception as exc:
        finished_at = now_iso()
        record_sync_run(
            conn,
            account_id=account_id,
            folder=folder,
            started_at=started_at,
            finished_at=finished_at,
            ok=False,
            fetched_count=fetched_count,
            stored_count=stored_count,
            error_text=str(exc),
        )
        raise

    finished_at = now_iso()
    record_sync_run(
        conn,
        account_id=account_id,
        folder=folder,
        started_at=started_at,
        finished_at=finished_at,
        ok=True,
        fetched_count=fetched_count,
        stored_count=stored_count,
        error_text="",
    )
    return {
        "ok": True,
        "db_path": str(db_path),
        "account_id": account_id,
        "folder": folder,
        "fetched_count": fetched_count,
        "stored_count": stored_count,
        "finished_at": finished_at,
    }


def smtp_send(args: argparse.Namespace) -> dict:
    password = require_password(args)
    db_path = Path(args.db)
    conn = connect_db(db_path)
    account_id = upsert_account(
        conn,
        email_address=args.email,
        imap_host=args.imap_host,
        imap_port=args.imap_port,
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
    )

    recipients = normalize_addresses(args.to)
    cc_emails = normalize_addresses(args.cc or [])
    if not recipients:
        raise SystemExit("Need at least one recipient in --to.")

    msg = EmailMessage()
    msg["From"] = args.email
    msg["To"] = ", ".join(recipients)
    if cc_emails:
        msg["Cc"] = ", ".join(cc_emails)
    msg["Subject"] = args.subject
    msg["Message-ID"] = make_msgid(domain=args.email.split("@", 1)[-1])
    msg.set_content(args.body)

    with smtplib.SMTP_SSL(args.smtp_host, args.smtp_port, context=ssl.create_default_context()) as smtp:
        smtp.login(args.email, password)
        smtp.send_message(msg, from_addr=args.email, to_addrs=recipients + cc_emails)

    upsert_account(
        conn,
        email_address=args.email,
        imap_host=args.imap_host,
        imap_port=args.imap_port,
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        mark_smtp_ok=True,
    )
    outbound = stored_message_from_outbound(
        account_id=account_id,
        sender_email=args.email,
        to_emails=recipients,
        cc_emails=cc_emails,
        subject=args.subject,
        body_text=args.body,
        message_id_header=msg["Message-ID"],
    )
    upsert_message(conn, outbound)
    conn.commit()
    return {
        "ok": True,
        "db_path": str(db_path),
        "account_id": account_id,
        "smtp_host": args.smtp_host,
        "smtp_port": args.smtp_port,
        "to": recipients,
        "subject": args.subject,
        "message_id": msg["Message-ID"],
    }


def list_cached(args: argparse.Namespace) -> dict:
    conn = connect_db(Path(args.db))
    rows = conn.execute(
        """
        SELECT account_id, folder, direction, subject, from_email, received_at_iso, preview
        FROM mail_messages
        ORDER BY received_at_ts DESC
        LIMIT ?
        """,
        (args.limit,),
    ).fetchall()
    messages = [
        {
            "account_id": row[0],
            "folder": row[1],
            "direction": row[2],
            "subject": row[3],
            "from_email": row[4],
            "received_at": row[5],
            "preview": row[6],
        }
        for row in rows
    ]
    return {"ok": True, "db_path": args.db, "count": len(messages), "messages": messages}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight IMAP/SMTP mail tool for the CTO-Agent.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--email", required=False, default=os.environ.get("CTO_EMAIL_ADDRESS", ""))
    parser.add_argument("--password", default="")
    parser.add_argument("--password-env", default="CTO_EMAIL_PASSWORD")
    parser.add_argument("--imap-host", default=DEFAULT_IMAP_HOST)
    parser.add_argument("--imap-port", type=int, default=DEFAULT_IMAP_PORT)
    parser.add_argument("--smtp-host", default=DEFAULT_SMTP_HOST)
    parser.add_argument("--smtp-port", type=int, default=DEFAULT_SMTP_PORT)

    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Sync a mailbox folder over IMAP into SQLite.")
    sync_parser.add_argument("--folder", default="INBOX")
    sync_parser.add_argument("--limit", type=int, default=20)
    sync_parser.add_argument("--mark-seen", action="store_true")
    sync_parser.set_defaults(func=imap_sync)

    send_parser = subparsers.add_parser("send", help="Send a mail over SMTP and cache the outbound record.")
    send_parser.add_argument("--to", nargs="+", required=True)
    send_parser.add_argument("--cc", nargs="*", default=[])
    send_parser.add_argument("--subject", required=True)
    send_parser.add_argument("--body", required=True)
    send_parser.set_defaults(func=smtp_send)

    list_parser = subparsers.add_parser("list", help="List cached mails from SQLite.")
    list_parser.add_argument("--limit", type=int, default=10)
    list_parser.set_defaults(func=list_cached)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.email and args.command in {"sync", "send"}:
        raise SystemExit("Missing email address. Set --email or export CTO_EMAIL_ADDRESS.")
    try:
        result = args.func(args)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
