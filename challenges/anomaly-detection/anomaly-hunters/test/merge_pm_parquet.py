"""Merge Parquet members of every _PM_ tar archive without filtering captures.

Run with .venv-pm-merge/bin/python merge_pm_parquet.py. The output is published
only after row counts, metadata, and every waveform value pass a read-back check.
"""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import time
import csv

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def fingerprint(path):
    value = path.stat()
    return dict(size_bytes=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino)


def metadata_json(schema):
    return {base64.b64encode(key).decode(): base64.b64encode(value).decode()
            for key, value in (schema.metadata or {}).items()}


def update_hashes(table, metadata_digest, waveform_digest):
    """Hash logical rows independently of Arrow batch and Parquet row boundaries."""
    metadata = table.drop(["data_array"])
    for index, field in enumerate(metadata.schema):
        if pa.types.is_timestamp(field.type):
            # Preserve nanoseconds without depending on pandas.Timestamp.
            metadata = metadata.set_column(index, field.name, metadata.column(index).cast(pa.int64()))
    for row in metadata.to_pylist():
        metadata_digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                          default=str, ensure_ascii=False).encode())
        metadata_digest.update(b"\n")
    for array in table.column("data_array").chunks:
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        valid = array.is_valid().to_numpy(zero_copy_only=False)
        values = array.values
        for row in range(len(array)):
            if not valid[row]:
                waveform_digest.update(b"NULL\n")
                continue
            first, last = int(offsets[row]), int(offsets[row+1])
            waveform = values.slice(first, last-first)
            waveform_digest.update(f"{waveform.type}:{len(waveform)}\n".encode())
            if waveform.null_count:
                mask = waveform.is_valid().to_numpy(zero_copy_only=False)
                waveform_digest.update(b"MASK"+mask.tobytes())
                numbers = pc.drop_null(waveform).to_numpy(zero_copy_only=False)
            else:
                waveform_digest.update(b"ALL_VALID")
                numbers = waveform.to_numpy(zero_copy_only=False)
            waveform_digest.update(memoryview(np.ascontiguousarray(numbers)).cast("B"))


def align_table(batch, schema, archive, member, first_row):
    table = pa.Table.from_batches([batch])
    size = len(table)
    provenance = {
        "source_archive": pa.array([str(archive)]*size, type=pa.large_string()),
        "source_parquet": pa.array([member]*size, type=pa.large_string()),
        "source_row": pa.array(np.arange(first_row, first_row+size, dtype=np.uint64)),
    }
    columns = []
    for field in schema:
        if field.name in provenance:
            column = provenance[field.name]
        elif field.name in table.column_names:
            column = table.column(field.name)
        else:
            column = pa.nulls(size, type=field.type)
        columns.append(column.cast(field.type, safe=True))
    return pa.Table.from_arrays(columns, schema=schema)


def merge(source, output, batch_size=128):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    partial = output.with_name(output.name+".partial")
    summary_path = output.with_suffix(".summary.json")
    inventory_path = output.with_suffix(".inputs.json")
    cycles_path = output.with_suffix(".cycle_counts.csv")
    pvs_path = output.with_suffix(".pv_counts.csv")
    for path in [output, partial, summary_path, inventory_path, cycles_path, pvs_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    archives = sorted(path for path in source.rglob("*")
                      if path.is_file() and "_PM_" in path.name and ".tar" in path.name)
    if not archives:
        raise ValueError("No _PM_ tar archives found")
    before = {str(path): fingerprint(path) for path in archives}
    pa.set_cpu_count(min(8, os.cpu_count() or 1))
    started = time.monotonic()
    def log(message):
        print(f"[{time.monotonic()-started:.1f}s] {message}", flush=True)
    log(f"Found {len(archives)} PM archives; merging all their Parquet members")
    inventory, schemas, actual_paths = [], [], {}
    duplicate_groups = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix=".pm-merge-", dir=output.parent) as temporary:
        for number, archive in enumerate(archives):
            key = re.search(r"^(.*_PM_\d+)", archive.name)
            duplicate_groups[key.group(1) if key else archive.name].append(str(archive))
            with archive.open("rb") as stream:
                compressed = stream.read(2) == b"\x1f\x8b"
            actual = archive
            if compressed:
                actual = Path(temporary)/f"archive_{number:04d}.tar"
                log(f"Decompressing {archive.name} once for streaming access")
                report_at = time.monotonic()
                with gzip.open(archive, "rb") as reader, actual.open("xb") as writer:
                    copied = 0
                    while True:
                        chunk = reader.read(16*1024*1024)
                        if not chunk:
                            break
                        writer.write(chunk)
                        copied += len(chunk)
                        if time.monotonic()-report_at >= 10:
                            log(f"Decompressed {copied/1024**3:.2f} GiB")
                            report_at = time.monotonic()
            actual_paths[str(archive)] = actual
            with tarfile.open(actual, "r:") as container:
                for member in container:
                    if not member.isfile() or not member.name.lower().endswith((".parquet", ".paquet")):
                        continue
                    with container.extractfile(member) as stream:
                        parquet = pq.ParquetFile(stream)
                        schema = parquet.schema_arrow
                        schemas.append(schema.remove_metadata())
                        inventory.append(dict(archive=str(archive), member=member.name,
                            member_bytes=member.size, rows=parquet.metadata.num_rows,
                            row_groups=parquet.num_row_groups,
                            schema=[dict(name=f.name, type=str(f.type), nullable=f.nullable) for f in schema],
                            schema_metadata_base64=metadata_json(schema)))
        if not inventory:
            raise ValueError("No Parquet files found inside PM archives")
        unified = pa.unify_schemas(schemas, promote_options="permissive")
        for name in ["cycle_id", "pv", "data_array"]:
            if name not in unified.names:
                raise ValueError(f"Missing required waveform column: {name}")
        provenance = [pa.field("source_archive", pa.large_string()),
                      pa.field("source_parquet", pa.large_string()), pa.field("source_row", pa.uint64())]
        if any(f.name in unified.names for f in provenance):
            raise ValueError("Source schema already contains merge provenance columns")
        schema = pa.schema(list(unified)+provenance, metadata={
            b"pm_merge:inputs": inventory_path.name.encode(),
            b"pm_merge:selection": b"All Parquet members in _PM_ archives; no row filtering or deduplication",
        })
        expected_rows = sum(item["rows"] for item in inventory)
        log(f"Inventory: {len(inventory)} Parquet files / {expected_rows:,} rows / {len(schema)} output columns")
        string_columns = [field.name for field in schema if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)]
        cycles, pvs, lengths, modes = Counter(), Counter(), Counter(), Counter()
        metadata_digest, waveform_digest = hashlib.sha256(), hashlib.sha256()
        rows_written, completed_members = 0, 0
        last_report = time.monotonic()
        try:
            with pq.ParquetWriter(partial, schema, compression="zstd", compression_level=1,
                                  use_dictionary=string_columns, write_statistics=True) as writer:
                for archive in archives:
                    entries = [item for item in inventory if item["archive"] == str(archive)]
                    with tarfile.open(actual_paths[str(archive)], "r:") as container:
                        for item in entries:
                            offset = 0
                            with container.extractfile(item["member"]) as stream:
                                parquet = pq.ParquetFile(stream)
                                for batch in parquet.iter_batches(batch_size=batch_size, use_threads=True):
                                    table = align_table(batch, schema, archive, item["member"], offset)
                                    update_hashes(table, metadata_digest, waveform_digest)
                                    writer.write_table(table, row_group_size=batch_size)
                                    cycles.update(table.column("cycle_id").to_pylist())
                                    pvs.update(table.column("pv").to_pylist())
                                    if "beam_mode" in table.column_names:
                                        modes.update(table.column("beam_mode").to_pylist())
                                    lengths.update(pc.list_value_length(table.column("data_array")).to_pylist())
                                    offset += len(table)
                                    rows_written += len(table)
                                    if time.monotonic()-last_report >= 10:
                                        log(f"Merged {rows_written:,}/{expected_rows:,} rows; {completed_members}/{len(inventory)} files complete")
                                        last_report = time.monotonic()
                            if offset != item["rows"]:
                                raise ValueError(f"Source row-count mismatch: {archive.name}/{item['member']}")
                            completed_members += 1
                    log(f"Archive complete {archive.name}: {rows_written:,} total rows")
            if rows_written != expected_rows:
                raise AssertionError("Merged row count differs from input inventory")
            log("Reading back the entire merged file to verify metadata and every waveform value")
            expected_metadata, expected_waveforms = metadata_digest.hexdigest(), waveform_digest.hexdigest()
            metadata_digest, waveform_digest = hashlib.sha256(), hashlib.sha256()
            restored = pq.ParquetFile(partial)
            if not restored.schema_arrow.equals(schema, check_metadata=True) or restored.metadata.num_rows != expected_rows:
                raise ValueError("Output schema or row count differs")
            checked = 0
            for batch in restored.iter_batches(batch_size=batch_size, use_threads=True):
                update_hashes(pa.Table.from_batches([batch]), metadata_digest, waveform_digest)
                checked += len(batch)
                if time.monotonic()-last_report >= 10:
                    log(f"Verified {checked:,}/{expected_rows:,} rows")
                    last_report = time.monotonic()
            if checked != expected_rows or metadata_digest.hexdigest() != expected_metadata or waveform_digest.hexdigest() != expected_waveforms:
                raise ValueError("Read-back verification failed: metadata or waveform values changed")
            if any(fingerprint(path) != before[str(path)] for path in archives):
                raise RuntimeError("A source archive changed during merging")
            summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), source=str(source), output=str(output),
                archives=len(archives), parquet_files=len(inventory), rows=expected_rows,
                cycles=len(cycles), pvs=len(pvs), columns=schema.names, output_bytes=partial.stat().st_size,
                waveform_lengths={"null" if key is None else str(key): value for key, value in lengths.items()},
                beam_modes={"null" if key is None else str(key): value for key, value in modes.items()},
                row_order="Archive filename order, member order within each archive, then original Parquet row order",
                duplicate_archive_event_groups={key: values for key, values in duplicate_groups.items() if len(values)>1},
                duplicate_policy="All rows from all source archives retained; no deduplication",
                filtering="None; all PVs and all waveform samples retained", provenance_columns=[f.name for f in provenance],
                metadata_sha256=expected_metadata, waveform_sha256=expected_waveforms,
                full_readback_verified=True, source_files_unchanged=True, pyarrow_version=pa.__version__,
                elapsed_seconds=time.monotonic()-started)
            with partial.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(partial, output)
            with inventory_path.open("x") as stream:
                json.dump(dict(archives=before, parquet_files=inventory), stream, indent=2)
                stream.write("\n")
            with summary_path.open("x") as stream:
                json.dump(summary, stream, indent=2)
                stream.write("\n")
            for path, counter, name in [(cycles_path, cycles, "cycle_id"), (pvs_path, pvs, "pv")]:
                with path.open("x", newline="") as stream:
                    out = csv.writer(stream)
                    out.writerow([name, "waveforms"])
                    out.writerows(sorted(counter.items(), key=lambda item: str(item[0])))
            log("SAVED "+json.dumps({key:summary[key] for key in ["output", "archives", "parquet_files", "rows", "cycles", "pvs", "output_bytes", "full_readback_verified"]}))
            return summary
        finally:
            if partial.exists():
                partial.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/lustre/work/pdl18409/shared/data/anomaly-detection/raw")
    parser.add_argument("--output", default="pm_all_cycles.parquet")
    parser.add_argument("--batch-size", type=int, default=128)
    merge(**vars(parser.parse_args()))
