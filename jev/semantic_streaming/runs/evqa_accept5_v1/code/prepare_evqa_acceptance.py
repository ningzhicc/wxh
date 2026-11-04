#!/usr/bin/env python3
"""Acquire five pinned EVQA videos using bounded ZIP ranges; keep answers separate."""
import argparse
import ast
import csv
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import qwen_roi as q
from devset import range_bytes
from pipeline_media import file_sha, probe
from prepare_semantic_rollout import seal


REVISION = '8194befd7e4f52071664a46033ae9415e0c7a939'
BASE = f'https://huggingface.co/datasets/Salesforce/ST-Evidence-Bench/resolve/{REVISION}'
ARCHIVE = {'url': BASE + '/st_evidence_gen/videos_6fps.zip', 'size': 8907114504,
           'upstream_lfs_sha256': '90d63a9c0f2f55786463426e295b3c45d27631e5fdbe7fadd9c499235438a50c'}
ENTRY_IDS = ['qa_ego4d_trimmed25', 'qa_ego4d_trimmed73', 'qa_ego4d_trimmed100', '32795_0', 'W0XPU_7f5351']
DATASET = 'datasets/evqa_accept5_v1'


class RemoteZipFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= self.position <= self.size:
            raise ValueError('Remote seek outside archive')
        return self.position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count > 16 * 1024**2:
            raise ValueError('Full archive download refused; remote reads are bounded')
        if count <= 0:
            return b''
        value = range_bytes(self.url, self.position, self.position + count - 1, self.size)
        self.position += len(value)
        return value


def metadata_file(dataset, name, url):
    path = dataset / 'provenance' / name
    receipt = path.with_suffix(path.suffix + '.receipt.json')
    if receipt.exists():
        if file_sha(path) != json.loads(receipt.read_text())['sha256']:
            raise ValueError('Pinned metadata cache changed')
        return path
    if path.exists():
        raise ValueError('Unrecorded metadata exists')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'wxh-evqa-acceptance/1.0'})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(4 * 1024**2)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    q.write_json(receipt, {'url': url, 'sha256': file_sha(path), 'bytes': len(data), 'retrieved_at': q.now()})
    return path


def verify_dataset(dataset):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if file_sha(dataset / name) != expected:
            raise ValueError('Acceptance metadata changed: ' + name)
    cases = [json.loads(l) for l in (dataset / 'cases.jsonl').read_text().splitlines()]
    for case in cases:
        if file_sha(q.local_path(case['video_path'])) != case['video_sha256']:
            raise ValueError('Acceptance video changed')
    return cases


def prepare():
    dataset = q.local_path(DATASET)
    if (dataset / 'manifest.json').exists():
        return verify_dataset(dataset)
    dataset.mkdir(exist_ok=True)
    csv_path = metadata_file(dataset, 'st_evidence_gen.csv', BASE + '/st_evidence_gen/st_evidence_gen.csv')
    metadata_file(dataset, 'upstream_README.md', BASE + '/README.md')
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding='utf-8-sig'))))
    selected = [next(r for r in rows if r['entry_id'] == entry) for entry in ENTRY_IDS]
    if len({r['video_path'] for r in selected}) != 5:
        raise ValueError('Five different source videos are required')
    trained = sorted({json.loads(p.read_text())['raw_sha256'] for p in (q.ROOT / 'datasets/pilot_dev10_v1/provenance/preparation').glob('*.json')})
    cases, answers = [], []
    with zipfile.ZipFile(RemoteZipFile(ARCHIVE['url'], ARCHIVE['size'])) as archive:
        infos = [archive.getinfo('videos_6fps/' + row['video_path']) for row in selected]
        if sum(info.compress_size for info in infos) > 200 * 1024**2:
            raise ValueError('Selected media exceed the 200 MiB download budget')
        for index, (row, info) in enumerate(zip(selected, infos, strict=True), 1):
            cid = f'evqa_{index:02d}'
            video = q.ROOT / 'media/raw/evqa_accept5_v1' / (cid + '.mp4')
            receipt_path = dataset / 'provenance' / (cid + '_download.json')
            if video.exists():
                receipt = json.loads(receipt_path.read_text())
                if file_sha(video) != receipt['sha256'] or video.stat().st_size != info.file_size:
                    raise ValueError('Previously downloaded media changed')
            else:
                video.parent.mkdir(parents=True, exist_ok=True)
                temp = video.with_suffix('.mp4.part')
                with archive.open(info) as source, temp.open('wb') as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                if temp.stat().st_size != info.file_size:
                    raise ValueError('ZIP member size mismatch')
                temp.replace(video)
                q.write_json(receipt_path, {'member': info.filename, 'zip_crc32': f'{info.CRC:08x}',
                    'compressed_bytes': info.compress_size, 'bytes': info.file_size, 'sha256': file_sha(video),
                    'method': 'bounded exact HTTP ranges and ZIP CRC verification; full ZIP SHA not checked locally',
                    'archive': ARCHIVE, 'retrieved_at': q.now()})
            source_hash = file_sha(video)
            if source_hash in trained:
                raise ValueError('Selected video overlaps local finetuning source bytes')
            media_info = dataset / 'provenance' / cid / 'media_info.json'
            if media_info.exists():
                media = json.loads(media_info.read_text())
            else:
                media = probe(q.ROOT / '.tools/ffmpeg', video, media_info.parent, 'source')
                q.write_json(media_info, {k: v for k, v in media.items() if k != 'pixel_hashes'})
            duration = min(15.0, media['duration_s'])
            segments = ast.literal_eval(row['segment'])
            if any(not 0 <= start < end <= duration for start, end in segments):
                raise ValueError('Selected window does not contain complete annotated evidence')
            options = ast.literal_eval(row['candidates'])
            if len(options) < 2 or options.count(row['answer']) != 1:
                raise ValueError('Ambiguous QA options')
            cases.append({'case_id': cid, 'entry_id': row['entry_id'], 'question': row['question'],
                'video_path': str(video.relative_to(q.ROOT)), 'video_sha256': source_hash,
                'source': row['source'], 'source_video_id': row['video_id'], 'source_video_path': row['video_path'],
                'start_s': 0, 'duration_s': duration, 'published_fps': media['fps'],
                'not_in_local_finetuning_media_by_sha256': True, 'qa_choices': options})
            answers.append({'case_id': cid, 'reference_answer': row['answer'], 'reference_choice_index': options.index(row['answer']),
                'evidence_segments_s': segments, 'local_human_verified': False,
                'source_csv_row_sha256': q.sha(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())})
            q.write_json(dataset / 'provenance' / (cid + '_upstream_row.json'), row)
            print(f"Prepared {cid}: {row['question']} ({duration:.3f}s at {media['fps']} fps)", flush=True)
    (dataset / 'cases.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in cases))
    (dataset / 'annotations').mkdir(exist_ok=True)
    (dataset / 'annotations/answers.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in answers))
    q.write_json(dataset / 'sources.lock.json', {'repo': 'Salesforce/ST-Evidence-Bench', 'revision': REVISION,
        'archive': ARCHIVE, 'csv_sha256': file_sha(csv_path), 'entry_ids': ENTRY_IDS,
        'local_finetuning_source_sha256': trained, 'license': 'CC-BY-NC-4.0; upstream declares research use',
        'spatial_masks_downloaded': False, 'full_archive_downloaded': False})
    (dataset / 'code').mkdir(exist_ok=True)
    (dataset / 'code/prepare_evqa_acceptance.py').write_bytes(Path(__file__).read_bytes())
    q.write_json(dataset / 'manifest.json', {'created_at': q.now(), 'case_count': 5,
        'role': 'small acceptance set excluded from local semantic finetuning; not a claim of pretraining/source-family disjointness',
        'metadata_only_exclusion_check': False, 'training_started': False, 'files_sha256': seal(dataset),
        'limits': 'Published 6fps media; 5 examples, not official full benchmark accuracy. Reference answers not locally human-verified. No spatial mask metric.'})
    return cases


if __name__ == '__main__':
    cases = prepare()
    print(json.dumps({'prepared_cases': len(cases), 'dataset': DATASET, 'api_calls': 0, 'gpu_used': False}))
