"""Frame-exact continuous CFR preparation and measured ROI encoding for the pipeline."""
from fractions import Fraction
import math
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image

import qwen_roi as q
import encode_roi_smoke as codec
import encode_roi_grid as grid


def file_sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frame_table(path):
    lines = Path(path).read_text().splitlines()
    tb = Fraction(next(x.split(':', 1)[1].strip() for x in lines if x.startswith('#tb 0:')))
    dimensions = next(x.split(':', 1)[1].strip() for x in lines if x.startswith('#dimensions 0:'))
    rows = [[v.strip() for v in line.split(',')] for line in lines if line and not line.startswith('#')]
    if len(rows) < 2:
        raise ValueError('At least two decoded frames are required')
    pts = [int(row[2]) for row in rows]
    steps = {b - a for a, b in zip(pts, pts[1:])}
    if len(steps) != 1 or next(iter(steps)) <= 0:
        raise ValueError('VFR input requires a timestamp-aware path; this pipeline supports CFR only')
    fps = 1 / (tb * next(iter(steps)))
    width, height = map(int, dimensions.split('x'))
    if width % 2 or height % 2:
        raise ValueError('Even image dimensions are required for yuv420p')
    return {'frames': len(rows), 'fps': str(fps), 'duration_s': float(len(rows) / fps),
            'width': width, 'height': height, 'first_pts_s': float(pts[0] * tb),
            'pixel_hashes': [row[-1] for row in rows]}


def probe(ffmpeg, video, directory, stem='probe', window=None):
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / (stem + '.framemd5')
    cmd = [str(ffmpeg), '-hide_banner', '-nostdin', '-n', '-v', 'error', '-threads', '2']
    if window:
        cmd += ['-ss', str(window[0])]
    cmd += ['-i', str(video)]
    if window:
        cmd += ['-t', str(window[1])]
    cmd += ['-map', '0:v:0', '-an', '-pix_fmt', 'yuv420p', '-threads', '1',
            '-fps_mode', 'passthrough', '-f', 'framemd5', str(out)]
    codec.command(cmd, directory / (stem + '.log'))
    return frame_table(out)


def partitions(count, fps, seconds):
    if count < 2 or seconds <= 0:
        raise ValueError('Invalid video length or chunk duration')
    step = math.ceil(Fraction(str(seconds)) * Fraction(fps))
    if step < 2:
        raise ValueError('A chunk must span at least two frames')
    spans = [(i, min(i + step, count)) for i in range(0, count, step)]
    if spans[-1][1] - spans[-1][0] == 1 and len(spans) > 1:
        spans[-2:] = [(spans[-2][0], count)]
    return spans


def sampling(spans, fps, cfg):
    fps = Fraction(fps)
    result = {}
    if cfg.get('global_sample_times_s') is not None:
        for t in cfg['global_sample_times_s']:
            index = math.ceil(Fraction(str(t)) * fps)
            if not 0 <= index < spans[-1][1]:
                raise ValueError('Global sampling timestamp outside the prepared window')
            result[index] = float(t)
    else:
        for start, end in spans:
            for t in cfg['sample_times_s']:
                local = math.ceil(Fraction(str(t)) * fps)
                if 0 <= local < end - start:
                    result[start + local] = float(start / fps) + float(t)
    for start, end in spans:
        if not any(start <= index < end for index in result):
            raise ValueError('Each chunk needs at least one ROI sample; adjust sampling timestamps')
    return result


def prepare(ffmpeg, video, media, run, cfg):
    logs = run / 'prepare'
    logs.mkdir()
    source = probe(ffmpeg, video, logs, 'source_window', (cfg['start_s'], cfg['duration_s']))
    fps = Fraction(source['fps'])
    reference = media / 'reference.mp4'
    cmd = [str(ffmpeg), '-hide_banner', '-nostdin', '-n', '-threads', '2', '-ss', str(cfg['start_s']),
        '-i', str(video), '-map', '0:v:0', '-an', '-sn', '-dn', '-vf', 'setpts=PTS-STARTPTS',
        '-c:v', 'libx264', '-qp', '0', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-threads', '2',
        '-frames:v', str(source['frames']), '-r', source['fps'], '-fps_mode', 'cfr', '-movflags', '+faststart', str(reference)]
    codec.command(cmd, logs / 'reference.log')
    ref = probe(ffmpeg, reference, logs, 'reference')
    if ref['pixel_hashes'] != source['pixel_hashes'] or ref['fps'] != source['fps']:
        raise ValueError('Prepared reference differs from the decoded source window')
    spans = partitions(ref['frames'], fps, cfg['chunk_s'])
    samples = sampling(spans, fps, cfg)
    chunks, inputs = [], []
    for number, (start, end) in enumerate(spans):
        cid = f'chunk_{number:04d}'
        directory = media / 'chunks' / cid
        directory.mkdir(parents=True)
        chunk_logs = logs / cid
        chunk_logs.mkdir()
        path = directory / 'reference.mp4'
        cmd = [str(ffmpeg), '-hide_banner', '-nostdin', '-n', '-threads', '2', '-i', str(reference),
            '-map', '0:v:0', '-an', '-sn', '-dn', '-vf', f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
            '-c:v', 'libx264', '-qp', '0', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-threads', '2',
            '-frames:v', str(end - start), '-r', ref['fps'], '-fps_mode', 'cfr', '-movflags', '+faststart', str(path)]
        codec.command(cmd, chunk_logs / 'cut.log')
        actual = probe(ffmpeg, path, chunk_logs, 'reference')
        if actual['pixel_hashes'] != ref['pixel_hashes'][start:end]:
            raise ValueError('Non-contiguous or modified chunk frames')
        indices = sorted(i - start for i in samples if start <= i < end)
        decoded_info, images = grid.decode_samples(ffmpeg, path, directory / 'samples', indices)
        frame_inputs = []
        for local, image in zip(indices, images, strict=True):
            global_index = start + local
            row = {'case_id': cid, 'frame_id': f'{cid}_frame{global_index:06d}',
                'path': str(image.relative_to(q.ROOT)), 'sha256': file_sha(image),
                'question': cfg['question'], 'timestamp_s': samples[global_index],
                'actual_window_timestamp_s': float(global_index / fps),
                'source_timestamp_s': cfg['start_s'] + source['first_pts_s'] + float(global_index / fps),
                'source_frame_index': global_index, 'local_frame_index': local,
                'width': ref['width'], 'height': ref['height']}
            frame_inputs.append(row)
            inputs.append(row)
        chunks.append({'case_id': cid, 'index': number, 'start_frame': start, 'end_frame': end,
            'start_window_s': float(start / fps), 'duration_s': float((end - start) / fps),
            'fps': ref['fps'], 'frames': end - start, 'width': ref['width'], 'height': ref['height'],
            'reference_path': str(path.relative_to(q.ROOT)), 'reference_sha256': file_sha(path),
            'sample_indices': indices, 'inputs': frame_inputs, 'source_pixel_hashes': actual['pixel_hashes'],
            'sample_decode_info': decoded_info})
    q.write_json(run / 'chunks.json', chunks)
    q.write_json(run / 'roi_inputs.json', inputs)
    q.write_json(run / 'preparation.json', {'source_window': {k: v for k, v in source.items() if k != 'pixel_hashes'},
        'reference_path': str(reference.relative_to(q.ROOT)), 'reference_sha256': file_sha(reference),
        'continuous_frame_partition_verified': True, 'chunk_count': len(chunks),
        'normalization': 'Native dimensions/CFR; decoded 8-bit yuv420p reference. Pixel identity verified against source decoded with the same normalization.',
        'original_compressed_bitstream_preserved': False})
    return chunks


def encoding_grid():
    return [(crf, offset) for crf in (24, 32, 40) for offset in ('0', '-1/10', '-1/5', '-3/10')] + [
        (crf, offset) for crf in (46, 50) for offset in ('0', '-1/5')]


def measured_quality_utility(metrics, has_roi, scale):
    value = metrics['sampled_api_visible_evidence_rgb_psnr_db'] if has_roi else metrics['sampled_full_frame_rgb_psnr_db']
    basis = 'API_evidence_PSNR' if has_roi else 'full_frame_PSNR_no_localized_evidence'
    if value is None:
        # Here the mask is known nonempty, so null means zero MSE, not absence.
        # Keep measured PSNR null (positive infinity); cap ONLY the finite utility.
        return 60 / scale, basis + '_zero_MSE_utility_capped_at_60dB'
    return value / scale, basis


def encode_one(ffmpeg, chunk, rois, box, crf, offset, directory, logs, cfg):
    directory.mkdir(parents=True)
    logs.mkdir(parents=True)
    out = directory / 'video.mp4'
    cmd = [str(ffmpeg), '-hide_banner', '-nostdin', '-n', '-threads', '2', '-i', str(q.local_path(chunk['reference_path'])),
        '-map', '0:v:0', '-an', '-sn', '-dn']
    if box and Fraction(offset) != 0:
        x1, y1, x2, y2 = box
        cmd += ['-vf', f'addroi=x={x1}:y={y1}:w={x2-x1}:h={y2-y1}:qoffset={offset}']
    cmd += ['-c:v', 'libx264', '-preset', cfg['encoder_preset'], '-pix_fmt', 'yuv420p',
        '-threads', str(cfg['encoder_threads']), '-crf', str(crf), '-aq-mode', '1', '-aq-strength', '1',
        '-x264-params', 'mbtree=0', '-g', '60', '-keyint_min', '30', '-frames:v', str(chunk['frames']),
        '-r', chunk['fps'], '-fps_mode', 'cfr', '-movflags', '+faststart', str(out)]
    codec.command(cmd, logs / 'encode.log')
    info, frames = grid.decode_samples(ffmpeg, out, directory / 'decoded', chunk['sample_indices'])
    if (info['frame_count'] != chunk['frames'] or info['fps'] != chunk['fps']
            or info['dimensions'] != f"{chunk['width']}x{chunk['height']}"):
        raise ValueError('Encoded candidate changed video geometry or cadence')
    metrics = grid.quality(frames, [q.local_path(x['path']) for x in chunk['inputs']],
                           box or (0, 0, chunk['width'], chunk['height']), rois)
    if box is None:
        metrics.update(sampled_static_roi_rgb_psnr_db=None, sampled_background_rgb_psnr_db=metrics['sampled_full_frame_rgb_psnr_db'])
    utility, basis = measured_quality_utility(metrics, box is not None, cfg['quality_scale_db'])
    return {'video_path': str(out.relative_to(q.ROOT)), 'video_sha256': file_sha(out),
        'size_bytes': out.stat().st_size, 'actual_bitrate_kbps': out.stat().st_size * 8 / chunk['duration_s'] / 1000,
        'duration_s': chunk['duration_s'], 'global_crf': crf, 'roi_qoffset': offset,
        'mode': 'roi' if box and Fraction(offset) != 0 else 'uniform', 'roi_box_pixels': box,
        'metrics': metrics, 'offline_quality_proxy': utility, 'quality_basis': basis,
        'frame_info': info}


def encode_chunks(ffmpeg, chunks, roi_results, media, run, cfg):
    by_frame = {r['frame_id']: r for r in roi_results}
    actions, results = {}, []
    for chunk in chunks:
        cid = chunk['case_id']
        rois = [by_frame[row['frame_id']] for row in chunk['inputs']]
        box = (codec.union_box(rois, chunk['width'], chunk['height'], cfg['roi_padding_pixels'], 16)
               if any(r['regions'] for r in rois) else None)
        chunk_actions, unique = [], {}
        for slot, (crf, requested_offset) in enumerate(encoding_grid()):
            offset = requested_offset if box else '0'
            key = (crf, offset)
            candidate_id = f'{cid}_crf{crf}_slot{slot:02d}'
            if key not in unique:
                directory = media / 'candidates' / candidate_id
                logs = run / 'encoding' / candidate_id
                unique[key] = encode_one(ffmpeg, chunk, rois, box, crf, offset, directory, logs, cfg)
                q.write_json(logs / 'result.json', unique[key])
            action = {**unique[key], 'case_id': cid, 'candidate_id': candidate_id,
                'action_id': f"a{chunk['index'] * 16 + slot:02d}", 'requested_roi_qoffset': requested_offset,
                'encoding_alias': unique[key]['video_path'] if requested_offset != offset else None,
                'roi_present': box is not None}
            chunk_actions.append(action)
            results.append(action)
        actions[cid] = chunk_actions
        q.write_json(run / 'actions.json', actions)
        print(f'Encoded {cid}: {len(unique)} actual files, 16 catalog slots, ROI={box}', flush=True)
    q.write_json(run / 'candidates.json', results)
    return actions


def receive(ffmpeg, selected, chunk, media, run):
    cid = chunk['case_id']
    directory = media / 'received' / cid
    directory.mkdir(parents=True)
    source = q.local_path(selected['video_path'])
    if source.stat().st_size != selected['size_bytes'] or file_sha(source) != selected['video_sha256']:
        raise ValueError('Selected video changed before delivery')
    dest = directory / 'video.mp4'
    shutil.copyfile(source, dest)
    if file_sha(dest) != selected['video_sha256']:
        raise ValueError('Receiver bytes differ from selected file')
    info, _ = grid.decode_samples(ffmpeg, dest, directory / 'decoded', chunk['sample_indices'])
    if info != selected['frame_info']:
        raise ValueError('Receiver full decoded frames differ from selected candidate')
    receipt = {'case_id': cid, 'source_path': selected['video_path'], 'received_path': str(dest.relative_to(q.ROOT)),
        'sha256': selected['video_sha256'], 'size_bytes': selected['size_bytes'], 'all_frames_decode_verified': True,
        'frame_info': info, 'transport': 'verified local file copy; timing is independently trace-replayed'}
    q.write_json(run / 'receiver' / f'{cid}.json', receipt)
    return receipt


def assemble(ffmpeg, chunks, receivers, media, run):
    playlist = media / 'received' / 'playlist.ffconcat'
    playlist.write_text('ffconcat version 1.0\n' + ''.join(f"file '{c['case_id']}/video.mp4'\n" for c in chunks))
    output = media / 'received' / 'playback.mp4'
    cmd = [str(ffmpeg), '-hide_banner', '-nostdin', '-n', '-f', 'concat', '-safe', '1', '-i', str(playlist),
           '-map', '0:v:0', '-an', '-c', 'copy', '-movflags', '+faststart', str(output)]
    codec.command(cmd, run / 'receiver' / 'assemble.log')
    actual = probe(ffmpeg, output, run / 'receiver', 'playback')
    expected = []
    for receiver in receivers:
        decoded = q.local_path(receiver['received_path']).parent / 'decoded' / 'decoded.framemd5'
        expected += frame_table(decoded)['pixel_hashes']
    if actual['pixel_hashes'] != expected:
        raise ValueError('Playback concatenation changed, dropped or repeated selected frames')
    return {'path': str(output.relative_to(q.ROOT)), 'sha256': file_sha(output), 'frames': actual['frames'],
            'fps': actual['fps'], 'duration_s': actual['duration_s'], 'selected_frame_sequence_verified': True}
