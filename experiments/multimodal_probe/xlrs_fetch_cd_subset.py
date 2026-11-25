#!/usr/bin/env python3
"""Extract selected XLRS-Bench-lite rows with bounded HTTP Range reads.

The upstream Arrow IPC streams put all text columns and the image offsets
before the image payload. Read those first, skip unrelated image bytes, and
copy selected images in their original encoding. No shard is downloaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import re
import struct
from threading import Lock

from PIL import Image
import pyarrow as pa
import requests
from urllib3.exceptions import HTTPError

REPO = "initiacms/XLRS-Bench-lite"
SUBSET = Path("datasets/XLRS-Bench-lite/cd_subset")
MANIFEST = "dev_cd_vqa.jsonl"
CAP = 3_000_000_000
CATEGORIES = {
    "change detection", "regional counting with change detection",
    "counting with change detection", "regional counting with changing detection",
    "counting with changing detection", "environmental change",
    "spatiotemporal reasoning", "spatiotemporal vqa",
}
SCHEMA = pa.schema([
    ("path", pa.string()), ("index", pa.int32()), ("question", pa.string()),
    ("multi-choice options", pa.list_(pa.string())), ("answer", pa.string()),
    ("category", pa.string()), ("l2-category", pa.string()),
    ("image", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
])
BINARY_OFFSETS, BINARY_DATA = 26, 27
REGEX_SHARD = re.compile(r"train/data-\d{5}-of-00074\.arrow")


def is_change(category: str) -> bool:
    return any(re.sub(r"[-\s]+", " ", part.strip().lower()).replace(
        "spatio temporal", "spatiotemporal") in CATEGORIES
        for part in category.split("/"))


class Budget:
    def __init__(self) -> None:
        self.transferred = 0
        self.lock = Lock()

    def reserve(self, length: int) -> None:
        with self.lock:
            if length < 0 or self.transferred + length >= CAP:
                raise ValueError("download would reach the strict 3 GB cap")
            self.transferred += length


class RemoteArrow:
    def __init__(self, session, url: str, size: int, budget: Budget):
        self.session, self.url, self.size, self.budget = session, url, size, budget

    def read(self, start: int, length: int) -> bytes:
        if length == 0:
            return b""
        end = start + length - 1
        if not 0 <= start <= end < self.size:
            raise ValueError("Arrow buffer is outside its shard")
        self.budget.reserve(length)
        try:
            with self.session.get(
                self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                stream=True, timeout=(12, 60),
            ) as response:
                if (response.status_code != 206
                    or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
                    or response.headers.get("Content-Encoding", "identity") != "identity"):
                    raise ValueError(f"server did not honor exact Arrow range (HTTP {response.status_code})")
                payload = response.raw.read(length)
        except (requests.RequestException, HTTPError) as error:
            raise ValueError(f"Arrow range request failed ({type(error).__name__})") from None
        if len(payload) != length:
            raise ValueError("short Arrow range response")
        return payload


def frame_length(prefix: bytes) -> int:
    marker, length = struct.unpack("<II", prefix)
    if marker != 0xFFFFFFFF or length > 1024 * 1024:
        raise ValueError("unsupported Arrow IPC frame")
    return length


def batch_layout(metadata: bytes) -> tuple[int, int, list[tuple[int, int]]]:
    """Read Arrow Message/RecordBatch flatbuffer fields (uncompressed IPC only)."""
    def value(fmt: str, offset: int) -> int:
        return struct.unpack_from(fmt, metadata, offset)[0]

    def field(table: int, index: int) -> int:
        vtable = table - value("<i", table)
        slot = 4 + 2 * index
        offset = value("<H", vtable + slot) if slot < value("<H", vtable) else 0
        return table + offset if offset else 0

    message = value("<I", 0)
    if not field(message, 1) or value("<B", field(message, 1)) != 3:
        raise ValueError("expected an Arrow RecordBatch")
    pointer = field(message, 2)
    batch = pointer + value("<I", pointer)
    if field(batch, 3):
        raise ValueError("compressed Arrow IPC is not supported")
    pointer = field(batch, 2)
    vector = pointer + value("<I", pointer)
    count = value("<I", vector)
    rows = value("<q", field(batch, 0))
    body_length = value("<q", field(message, 3))
    if count != 31 or not (0 < rows <= 2000 and body_length > 0):
        raise ValueError("unexpected XLRS Arrow batch layout")
    buffers = [struct.unpack_from("<qq", metadata, vector + 4 + 16 * i) for i in range(count)]
    if any(start < 0 or n < 0 or start + n > body_length for start, n in buffers):
        raise ValueError("invalid Arrow buffer extent")
    return rows, body_length, buffers


@dataclass
class Candidate:
    path: str
    row: dict
    spans: tuple[tuple[int, int], ...]


def index_shard(remote) -> tuple[list[Candidate], Counter]:
    prefix = remote.read(0, 8)
    schema_len = frame_length(prefix)
    pos = 8 + schema_len
    header = remote.read(8, schema_len + 8)
    schema = pa.ipc.read_schema(pa.BufferReader(prefix + header[:-8]))
    if not schema.equals(SCHEMA):
        raise ValueError("unexpected XLRS Arrow schema; stopped before reading any images")
    prefix = header[-8:]
    result, counts = [], Counter()
    while (length := frame_length(prefix)):
        body = pos + 8 + length
        rows, body_len, buf = batch_layout(remote.read(pos + 8, length))
        if body + body_len > remote.size:
            raise ValueError("Arrow body is outside its shard")
        image_data, image_len = buf[BINARY_DATA]
        if image_data > 4 * 1024 * 1024:
            raise ValueError("unexpectedly large Arrow metadata")
        meta = remote.read(body, image_data)

        def data(i):
            start, length = buf[i]
            if start + length > len(meta):
                raise ValueError("metadata buffer overlaps image payload")
            return memoryview(meta)[start:start + length]

        def ints(i):
            return [x[0] for x in struct.iter_unpack("<i", data(i))]

        def strings(i):
            offsets, values = ints(i), data(i + 1)
            return [bytes(values[a:b]).decode("utf-8") for a, b in zip(offsets, offsets[1:])]

        categories, l2 = strings(17), strings(20)
        counts.update(categories)
        keep = [i for i in range(rows) if is_change(categories[i]) or is_change(l2[i])]
        if keep:
            indices, paths, questions, answers = ints(4), strings(1), strings(6), strings(14)
            option_lists, options = ints(9), strings(11)
            img_lists, offsets = ints(23), ints(BINARY_OFFSETS)
            for i in keep:
                image_range = range(img_lists[i], img_lists[i + 1])
                spans = tuple((body + image_data + offsets[j], offsets[j + 1] - offsets[j])
                              for j in image_range)
                if len(spans) not in (1, 2) or any(
                    length <= 0 or start < body + image_data or start + length > body + image_data + image_len
                    for start, length in spans
                ):
                    raise ValueError("target row has an invalid/unsupported embedded image layout")
                choice = answers[i].strip().upper()
                row = dict(id=f"xlrs-{indices[i]}", category=categories[i], question=questions[i],
                           options=options[option_lists[i]:option_lists[i + 1]], answer=choice)
                if not row["question"] or not 2 <= len(row["options"]) <= 4 or not re.fullmatch("[A-D]+", choice):
                    raise ValueError("target row is not multiple-choice VQA")
                result.append(Candidate(paths[i], row, spans))
        pos = body + body_len
        prefix = remote.read(pos, 8)
    return result, counts


def fetch(endpoint: str, output: Path, count: int) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("destination is not empty; refusing to overwrite an existing subset")
    budget = Budget()
    with requests.Session() as session:
        # Metadata API response is independently bounded before JSON parsing.
        try:
            with session.get(f"{endpoint}/api/datasets/{REPO}/tree/main/train",
                             stream=True, timeout=(10, 20)) as r:
                if r.status_code != 200:
                    raise ValueError(f"HF tree metadata returned HTTP {r.status_code}")
                r.raw.decode_content = True
                payload = r.raw.read(512 * 1024 + 1)
        except (requests.RequestException, HTTPError) as error:
            raise ValueError(f"HF tree metadata request failed ({type(error).__name__})") from None
    budget.reserve(len(payload))
    if len(payload) > 512 * 1024:
        raise ValueError("HF tree metadata exceeded the allowed size")
    files = {item["path"]: item["size"] for item in json.loads(payload)
             if REGEX_SHARD.fullmatch(item.get("path", "")) and item.get("type") == "file"}
    if len(files) != 74:
        raise ValueError(f"expected 74 source shard entries, found {len(files)}")

    def open_shard(session, path):
        return RemoteArrow(session, f"{endpoint}/datasets/{REPO}/resolve/main/{path}", files[path], budget)

    def inspect(path):
        with requests.Session() as session:
            selected, counts = index_shard(open_shard(session, path))
        return path, selected, counts

    selected, ids, scenes, repeats = [], set(), set(), []
    shards = sorted(files, reverse=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for start in range(0, len(shards), 6):
            for path, entries, found in pool.map(inspect, shards[start:start + 6]):
                print(f"index {path}: categories={dict(found)} target_rows={len(entries)}", flush=True)
                for item in entries:
                    if item.row["id"] in ids:
                        continue
                    ids.add(item.row["id"])
                    if item.path in scenes:
                        repeats.append((path, item))
                    else:
                        scenes.add(item.path)
                        selected.append((path, item))
            if len(selected) >= count:
                break
    selected = (selected + repeats)[:count]
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} target VQA rows found; no imagery downloaded")
    spans = {(path, span) for path, row in selected for span in row.spans}
    if budget.transferred + sum(span[1] for _, span in spans) >= CAP:
        raise ValueError("selected image payload would reach the strict 3 GB cap")
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "images"
    assets.mkdir(exist_ok=False)
    rows, stored = [], {}
    suffix = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif", "WEBP": ".webp", "BMP": ".bmp"}
    with requests.Session() as session:
        for path, item in selected:
            images, sizes = [], []
            for i, span in enumerate(item.spans, 1):
                key = (path, span)
                if key not in stored:
                    payload = open_shard(session, path).read(*span)
                    # Inspect headers only. Never resample or transcode the source.
                    if Image.MAX_IMAGE_PIXELS is not None:
                        Image.MAX_IMAGE_PIXELS = 150_000_000  # HRSCD can exceed Pillow's default warning.
                    with Image.open(io.BytesIO(payload)) as image:
                        file = assets / f"{item.row['id']}-{i}{suffix[image.format]}"
                        size = image.size
                    file.write_bytes(payload)
                    stored[key] = (file.relative_to(output).as_posix(), size)
                name, size = stored[key]
                images.append(name)
                sizes.append(size)
            rows.append(dict(item.row, images=images, image_sizes=sizes))
            print(f"extracted {item.row['id']}: {item.row['category']}, sizes={sizes}", flush=True)
    manifest = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    file_bytes = sum(p.stat().st_size for p in assets.iterdir()) + len(manifest)
    if file_bytes >= CAP:
        raise ValueError("subset would reach 3 GB on disk")
    (output / MANIFEST).write_bytes(manifest)
    print(f"source={REPO} rows={len(rows)} categories={dict(Counter(row['category'] for row in rows))} "
          f"images_per_row={dict(Counter(len(row['images']) for row in rows))} "
          f"image_shards={len({p for p, _ in selected})} downloaded_bytes={budget.transferred} "
          f"file_bytes={file_bytes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--output", type=Path, default=SUBSET)
    parser.add_argument("--count", type=int, choices=range(20, 31), default=24)
    args = parser.parse_args()
    try:
        fetch(args.endpoint.rstrip("/"), args.output, args.count)
    except (ValueError, OSError, KeyError, struct.error, pa.ArrowException) as error:
        # Transport errors above are normalized so redirects/proxy credentials are never logged.
        raise SystemExit(f"XLRS extraction stopped: {error}") from None


if __name__ == "__main__":
    main()
