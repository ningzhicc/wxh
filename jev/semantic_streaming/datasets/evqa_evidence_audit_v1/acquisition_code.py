#!/usr/bin/env python3
"""Offline-only EVQA mask audit. Never imported by inference or training."""
import sys,zipfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_evqa_acceptance import RemoteZipFile,BASE
from devset import range_bytes
import qwen_roi as q
class Cached(RemoteZipFile):
 def read(self,count=-1):
  count=self.size-self.position if count<0 else min(count,self.size-self.position)
  if count>32*1024**2:raise ValueError('read cap')
  out=bytearray()
  while count:
   block=self.position//(4*1024**2);start=block*4*1024**2;end=min(start+4*1024**2,self.size)-1
   p=q.ROOT/'datasets/evqa_evidence_audit_v1/provenance/ranges'/f'{start}_{end}.bin';r=p.with_suffix('.json')
   if r.exists():
    data=p.read_bytes()
    if q.sha(data)!=json.loads(r.read_text())['sha256']:raise ValueError('range changed')
   else:
    print('range',start,end,flush=True);data=range_bytes(self.url,start,end,self.size);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);q.write_json(r,{'url':self.url,'start':start,'end':end,'total':self.size,'sha256':q.sha(data),'retrieved_at':q.now()})
   pos=self.position-start;n=min(count,len(data)-pos);out.extend(data[pos:pos+n]);count-=n;self.position+=n
  return bytes(out)

import argparse
from PIL import Image, ImageDraw
from pipeline_media import file_sha
from run_semantic_pipeline import seal_paths, verify_files

ARCHIVE_SHA = '774d67a5919defd6d85a396075c051d694541cb07db17405125b251b91de99a8'
PAIRS = [('evqa_01', 'qa_ego4d_trimmed25'), ('evqa_02', 'qa_ego4d_trimmed73')]


def acquire():
 root=q.ROOT/'datasets/evqa_evidence_audit_v1'
 if (root/'files.sha256.json').exists():
  verify_files(json.loads((root/'files.sha256.json').read_text()));return root
 with zipfile.ZipFile(Cached(BASE+'/st_evidence_gen/masks.zip',842089918)) as z:
  for cid,eid in PAIRS:
   receipts=[]
   for info in z.infolist():
    if not info.filename.startswith('masks/'+eid+'/') or info.is_dir():continue
    raw=z.read(info);target=root/info.filename;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_bytes()!=raw:raise ValueError('Downloaded mask changed')
    target.write_bytes(raw);receipts.append({'member':info.filename,'bytes':len(raw),'sha256':q.sha(raw),'crc32':info.CRC,'header_offset':info.header_offset})
   q.write_json(root/'provenance'/(eid+'_members.json'),receipts)
  for name in ('masks/summary.json','masks/manifest.json'):
   info=z.getinfo(name)
   if info.file_size>8*1024**2:raise ValueError('Oversized metadata')
   raw=z.read(info);target=root/'provenance'/name.split('/')[-1]
   if target.exists() and target.read_bytes()!=raw:raise ValueError('Metadata changed')
   target.write_bytes(raw);q.write_json(target.with_suffix('.receipt.json'),{'member':name,'sha256':q.sha(raw),'bytes':len(raw),'crc32':info.CRC})
 q.write_json(root/'manifest.json',{'created_at':q.now(),'kind':'offline_only_spatial_evidence_audit',
  'upstream_url':BASE+'/st_evidence_gen/masks.zip','upstream_lfs_sha256':ARCHIVE_SHA,
  'full_archive_sha256_locally_verified':False,'member_crc_and_sha256_verified':True,
  'entry_ids':[eid for _,eid in PAIRS], 'source_dataset_manifest_sha256':file_sha(q.ROOT/'datasets/evqa_accept5_v1/manifest.json'),
  'alignment_warning':'Mask-export frame counts 183/185 differ from published video 180/180. Direct numeric-index correspondence is provisional; visual spot checks at 6,7,8s match objects. Empty masks may be black-filled, not verified absence. Not official EVQA mask metrics.'})
 (root/'acquisition_code.py').write_bytes(Path(__file__).read_bytes())
 q.write_json(root/'files.sha256.json',seal_paths([root]));return root


def coverage(mask, regions, width, height):
 binary=mask.convert('L').point(lambda v:255 if v else 0)
 binary=binary.resize((width,height),Image.Resampling.NEAREST)
 predicted=Image.new('1',(width,height));draw=ImageDraw.Draw(predicted)
 for r in regions:
  x1,y1,x2,y2=r['bbox_pixels'];draw.rectangle((x1,y1,x2-1,y2-1),fill=1)
 import numpy as np
 gt=np.array(binary)>0;pred=np.array(predicted)>0
 area=int(gt.sum());intersection=int((gt & pred).sum());union=int((gt | pred).sum())
 return {'mask_nonempty':bool(area),'gt_pixels':area,'gt_pixel_recall':intersection/area if area else None,
  'rectangle_mask_iou':intersection/union if area and union else None, 'api_regions':len(regions)}


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
 p.add_argument('--v2-study');a=p.parse_args();root=acquire();out=q.local_path('runs/'+a.run_id)
 if out.exists():raise ValueError('Use a new audit run ID')
 out.mkdir();q.write_json(out/'manifest.json',{'created_at':q.now(),'mask_seal_sha256':file_sha(root/'files.sha256.json'),
  'code_sha256':file_sha(Path(__file__)),'v2_study':a.v2_study,'offline_only':True})
 cases=[];ann={r['case_id']:r for r in map(json.loads,(q.ROOT/'datasets/evqa_accept5_v1/annotations/answers.jsonl').read_text().splitlines())}
 inherited=json.loads(q.local_path('runs/'+a.v2_study+'/policy_results.json').read_text()) if a.v2_study else {}
 export={r['entry_id']:r for r in json.loads((root/'provenance/manifest.json').read_text())}
 for cid,eid in PAIRS:
  run=q.ROOT/'runs'/('evqa_accept5_v1_'+cid+'_bandwidth');verify_files(json.loads((run/'files.sha256.json').read_text()))
  inputs=json.loads((run/'roi_inputs.json').read_text());variants={'v1':{r['frame_id']:r for r in map(json.loads,(run/'roi_results.jsonl').read_text().splitlines())}}
  if cid in inherited:
   v2=q.local_path('runs/'+inherited[cid]['source_run']);verify_files(json.loads((v2/'files.sha256.json').read_text()))
   other=json.loads((v2/'roi_inputs.json').read_text())
   if [(r['sha256'],r['timestamp_s'],r['question']) for r in inputs]!=[(r['sha256'],r['timestamp_s'],r['question']) for r in other]:raise ValueError('Audit images/questions/time differ')
   variants['v2']={r['frame_id']:r for r in map(json.loads,(v2/'roi_results.jsonl').read_text().splitlines())}
  rows=[]
  for row in inputs:
   mask=Image.open(root/'masks'/eid/f"{row['source_frame_index']:05d}.png").convert('L')
   item={'frame_id':row['frame_id'],'timestamp_s':row['actual_window_timestamp_s'],'provisional_mask_index':row['source_frame_index'],
    'in_annotated_time_segment':any(lo<=row['actual_window_timestamp_s']<=hi for lo,hi in ann[cid]['evidence_segments_s']), 'variants':{}}
   for version,rois in variants.items():
    item['variants'][version]=coverage(mask,rois[row['frame_id']]['regions'],row['width'],row['height'])
    # Index-shift sensitivity is a diagnostic, never a best-shift chosen for score.
    item['variants'][version]['recall_at_index_shifts']={}
    for delta in (-3,-1,0,1,3):
     mp=root/'masks'/eid/f"{row['source_frame_index']+delta:05d}.png"
     if mp.exists():item['variants'][version]['recall_at_index_shifts'][str(delta)]=coverage(Image.open(mp),rois[row['frame_id']]['regions'],row['width'],row['height'])['gt_pixel_recall']
   rows.append(item)
  scores={}
  for version in variants:
   valid=[r['variants'][version] for r in rows if r['in_annotated_time_segment'] and r['variants'][version]['mask_nonempty']]
   scores[version]={'nonempty_mask_samples_in_GT_time':len(valid),'empty_API_among_these':sum(r['api_regions']==0 for r in valid),
    'mean_GT_pixel_recall_provisional':sum(r['gt_pixel_recall'] for r in valid)/len(valid) if valid else None}
  cases.append({'case_id':cid,'entry_id':eid,'annotation':ann[cid],'export_alignment_metadata':export[eid],'scores':scores,'frames':rows})
 q.write_json(out/'audit.json',{'cases':cases,'caveat':json.loads((root/'manifest.json').read_text())['alignment_warning']})
 lines=['# 两个异常案例的离线证据核查','','官方掩码只用于本报告；原问题、原标准答案和旧结果均未修改。',
  '掩码导出与发布视频帧数不一致，以下按数字帧号对应，仅为诊断指标；空掩码可能是补黑，不能当成确认无物体。',
  '已目视核对 6、7、8 秒：碗的掩码对应右侧沥水架的浅蓝碗，毛巾对应洗手台下方的深色毛巾。', '',
  '| 案例 | 版本 | GT时间内非空掩码样本 | API空框 | 掩码像素覆盖率（暂定对齐） |','|---|---|---:|---:|---:|']
 for c in cases:
  for version,m in c['scores'].items():lines.append(f"| {c['case_id']} | {version} | {m['nonempty_mask_samples_in_GT_time']} | {m['empty_API_among_these']} | {m['mean_GT_pixel_recall_provisional']:.3f} |")
 lines+=['','碗案例存在多个可见碗，单帧定位会切换目标或漏掉右侧碗。毛巾案例原视频问答也与参考答案不一致，不能归因于编码损失；保留原标注 green，不按模型输出 blue 修改。',
  '完整窗口QA作为主比较；GT时间窗和掩码不输入定位、Kev或主QA。上述覆盖率不是官方J&F分数。']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');(out/'audit_code.py').write_bytes(Path(__file__).read_bytes());q.write_json(out/'files.sha256.json',seal_paths([out]));print(json.dumps({'audit':str(out),'cases':[{k:c[k] for k in ('case_id','scores')} for c in cases]},ensure_ascii=False))

if __name__=='__main__':main()
