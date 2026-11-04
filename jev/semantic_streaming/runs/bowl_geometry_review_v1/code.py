#!/usr/bin/env python3
"""Offline native-frame comparison: API boxes, official masks, encoder rectangles."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw
import qwen_roi as q
from pipeline_media import file_sha
from run_semantic_pipeline import seal_paths,verify_files
from audit_evqa_evidence import coverage


def build(run_id):
    source=q.local_path('runs/evqa_disambiguation_v1_bowl_bandwidth')
    masks=q.local_path('datasets/evqa_evidence_audit_v1')
    verify_files(json.loads((source/'files.sha256.json').read_text()))
    verify_files(json.loads((masks/'files.sha256.json').read_text()))
    out=q.local_path('runs/'+run_id);media=q.local_path('artifacts/'+run_id)
    if out.exists():raise ValueError('Use a new audit ID')
    out.mkdir();media.mkdir()
    inputs=json.loads((source/'roi_inputs.json').read_text());roi={r['frame_id']:r for r in map(json.loads,(source/'roi_results.jsonl').read_text().splitlines())}
    actions=json.loads((source/'actions.json').read_text());rows=[];sheets=[]
    for row in inputs:
        if not 7.309<=row['actual_window_timestamp_s']<=12.513:continue
        image=Image.open(q.local_path(row['path'])).convert('RGB')
        maskpath=masks/'masks/qa_ego4d_trimmed25'/f"{row['source_frame_index']:05d}.png";mask=Image.open(maskpath).convert('L')
        if mask.size!=image.size:raise ValueError('Native dimensions mismatch; do not silently resize audit')
        api=roi[row['frame_id']]['regions'];encoder=actions[row['case_id']][0]['roi_box_pixels'];bbox=mask.getbbox()
        # Fixed area chosen to show the rack, not fitted to improve overlap.
        crop=(400,60,700,300)
        raw=image.crop(crop).resize((600,480),Image.Resampling.NEAREST)
        marked=image.copy();overlay=Image.new('RGB',image.size,'#00d060')
        alpha=mask.point(lambda v:65 if v else 0);marked=Image.composite(overlay,marked,alpha)
        draw=ImageDraw.Draw(marked)
        if bbox:draw.rectangle((bbox[0],bbox[1],bbox[2]-1,bbox[3]-1),outline='#00ff80',width=2)
        for box in api:
            x1,y1,x2,y2=box['bbox_pixels'];draw.rectangle((x1,y1,x2-1,y2-1),outline='#ff2040',width=2)
        x1,y1,x2,y2=encoder;draw.rectangle((x1,y1,x2-1,y2-1),outline='#ffb020',width=2)
        marked=marked.crop(crop).resize((600,480),Image.Resampling.NEAREST)
        full=image.copy();full.thumbnail((480,360));panel=Image.new('RGB',(1680,535),'#131313')
        panel.paste(full,(0,60));panel.paste(raw,(480,55));panel.paste(marked,(1080,55))
        text=ImageDraw.Draw(panel);text.text((10,10),f"t={row['actual_window_timestamp_s']:.6f}s | frame={row['source_frame_index']} | original / native zoom / all regions",fill='white')
        text.text((500,32),'Raw pixels: 2x nearest zoom',fill='white');text.text((1090,32),'Red=API; green=official mask; orange=encoder rectangle',fill='white')
        path=media/(row['frame_id']+'.png');panel.save(path);sheets.append(path)
        metric=coverage(mask,api,image.width,image.height);protect=coverage(mask,[{'bbox_pixels':encoder}],image.width,image.height)
        rows.append({'frame_id':row['frame_id'],'source_frame_index':row['source_frame_index'],'timestamp_s':row['actual_window_timestamp_s'],
            'native_image':row['path'],'native_image_sha256':row['sha256'],'mask_path':str(maskpath.relative_to(q.ROOT)),'mask_sha256':file_sha(maskpath),
            'official_mask_bbox':bbox,'api_boxes':[r['bbox_pixels'] for r in api],'encoder_rectangle':encoder,
            'API_mask_recall':metric['gt_pixel_recall'],'encoder_mask_recall':protect['gt_pixel_recall'],
            'comparison_image':str(path.relative_to(q.ROOT)),'dimensions':list(image.size)})
    contact=Image.new('RGB',(1680,535*len(sheets)))
    for i,path in enumerate(sheets):contact.paste(Image.open(path),(0,535*i))
    contact.save(media/'contact.png')
    q.write_json(out/'audit.json',{'source_seal_sha256':file_sha(source/'files.sha256.json'),'mask_seal_sha256':file_sha(masks/'files.sha256.json'),
        'frames':rows,'crop_xyxy':[400,60,700,300],'GT_is_offline_only':True,'alignment_assumed':'Numeric index only; not declared corrected',
        'original_gate_recomputed':False,'original_gate_remains_failed':True})
    (out/'code.py').write_bytes(Path(__file__).read_bytes())
    # The visual assessment is saved as a separate, explicitly assistant-authored record.
    print(json.dumps({'run':str(out),'contact':str(media/'contact.png'),'frames':[{'t':r['timestamp_s'],'path':r['comparison_image']} for r in rows]},ensure_ascii=False))
    return out,media

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);a=p.parse_args();build(a.run_id)
