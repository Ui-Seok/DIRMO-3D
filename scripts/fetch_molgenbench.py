"""Fetch pinned official MolGenBench reference inputs, never published model outputs.

HTTP byte ranges avoid downloading gigabytes of other methods' results. Every
member is checked against the official ZIP CRC and recorded with SHA256. The
published whole-archive MD5 is recorded, NOT claimed verified for selective fetch.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import struct
import time
import urllib.request
import zipfile
import zlib
from zoneinfo import ZoneInfo

URL = 'https://zenodo.org/api/records/20233505/files/MolGenBench_Version4.zip/content'
SIZE = 5381684964
MD5 = '08739fa1d1c2d1709f1dbcc0886f7386'
COMMIT = 'f7b6cec1892cb575dc3adee4f5a71ee16ef04c39'

def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def fetch(start, end):
    for attempt in range(6):
        try:
            request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end-1}'})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end-1}/{SIZE}':
                    raise ValueError('Server did not honor exact byte range')
                data = response.read()
            if len(data) != end - start:
                raise ValueError('Incomplete byte range')
            return data
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt * 2, 60))

class RemoteZIP(io.RawIOBase):
    def __init__(self, cache):
        self.position = 0
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else SIZE)
        return self.position
    def read(self, n=-1):
        end = SIZE if n < 0 else min(SIZE, self.position + n)
        start = self.position
        self.position = end
        path = self.cache / f'{start}-{end}.bin'
        if not path.exists():
            ranges = [(a, min(a + 2**20 * 4, end)) for a in range(start, end, 2**20 * 4)]
            with ThreadPoolExecutor(max_workers=6) as pool:
                data = b''.join(pool.map(lambda ab: fetch(*ab), ranges))
            path.write_bytes(data)
        return path.read_bytes()

def wanted(name):
    path = PurePosixPath(name)
    if name.endswith('/') or '..' in path.parts or path.is_absolute(): return False
    base = path.name
    if 'reference_active_molecules' in path.parts:
        return (base.endswith('_reference_active_molecules.sdf') or
                base.endswith('_reference_ligand_pose_with_h.sdf') or
                base == 'top5_common_scaffold_info.csv')
    return base.endswith(('_prep.pdb', '_pocket10.pdb', '_lig.sdf'))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(RemoteZIP(root / 'zip-index-cache'))
    entries = sorted([x for x in archive.infolist() if wanted(x.filename)], key=lambda x:x.header_offset)
    dump(root / 'download_status.json', dict(status='fetching', time_kst=now(), files=len(entries)))
    # Consecutive entries are fetched together to keep requests below server limits.
    groups = []
    for item in entries:
        start = item.header_offset
        end = start + 30 + len(item.filename.encode('utf-8')) + 65535 + item.compress_size
        if groups and start - groups[-1][1] < 65536 and end - groups[-1][0] < 8 * 2**20:
            groups[-1][1] = max(groups[-1][1], end); groups[-1][2].append(item)
        else:
            groups.append([start, min(end, SIZE), [item]])
    records = []
    def extract(group):
        start, end, members = group
        if all((root / x.filename).exists() and
               zlib.crc32((root / x.filename).read_bytes()) == x.CRC for x in members):
            blob = None
        else:
            blob = fetch(start, end)
        output = []
        for item in members:
            dest = root / item.filename
            if blob is not None:
                header = item.header_offset - start
                if blob[header:header+4] != b'PK\x03\x04': raise ValueError('Invalid ZIP member')
                name_len, extra_len = struct.unpack_from('<HH', blob, header + 26)
                begin = header + 30 + name_len + extra_len
                data = blob[begin:begin+item.compress_size]
                if item.compress_type == zipfile.ZIP_DEFLATED: data = zlib.decompress(data, -15)
                elif item.compress_type != zipfile.ZIP_STORED: raise ValueError('Unsupported ZIP compression')
                if len(data) != item.file_size or zlib.crc32(data) != item.CRC:
                    raise ValueError(f'ZIP integrity mismatch: {item.filename}')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.with_suffix(dest.suffix+'.part').write_bytes(data)
                dest.with_suffix(dest.suffix+'.part').replace(dest)
            output.append(dict(path=item.filename, bytes=item.file_size, crc32=item.CRC,
                               sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        return output
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(extract, g) for g in groups]
        for future in as_completed(futures):
            records.extend(future.result())
            dump(root / 'download_status.json', dict(status='fetching', time_kst=now(), files=len(entries), completed=len(records)))
    dump(root / 'sources.json', dict(time_kst=now(), zenodo_record=20233505, version=4,
        url=URL, archive_bytes=SIZE, expected_archive_md5=MD5, whole_archive_md5_verified=False,
        integrity='Official ZIP central-directory CRC32 per member; local SHA256 recorded',
        official_code_commit=COMMIT, files=sorted(records, key=lambda r:r['path'])))
    dump(root / 'download_status.json', dict(status='complete', time_kst=now(), files=len(entries), completed=len(records)))
    print(json.dumps(dict(status='complete', files=len(records), bytes=sum(r['bytes'] for r in records))))

if __name__ == '__main__': main()
