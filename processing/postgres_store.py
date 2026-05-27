from __future__ import annotations

from typing import Sequence

import psycopg

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS lecture_chunks (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NULL,
    page INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


UPSERT_SQL = """
INSERT INTO lecture_chunks (source, page, chunk_index, text)
VALUES (%s, %s, %s, %s)
"""


def save_chunk_records(database_url: str, records: Sequence[dict[str, object]]) -> int:
    if not records:
        return 0

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_TABLE_SQL)
            params = []
            for record in records:
                text = str(record.get("text") or "")
                # remove NUL bytes which Postgres text type does not accept
                if "\x00" in text:
                    text = text.replace("\x00", "")
                if not text.strip():
                    # skip empty text chunks
                    continue
                params.append((
                    str(record["source"]),
                    int(record["page"]),
                    int(record["chunk_index"]),
                    text,
                ))
            if params:
                print(f"Saving {len(params)} chunk records to PostgreSQL")
                cursor.executemany(UPSERT_SQL, params)
        connection.commit()

    return len(records)