"""Derive unreviewed per-proposal measurements from existing pose traces."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from swisher.analysis import _metrics


def export_metrics(directory: Path) -> list[dict]:
    report=json.loads((directory/'report.json').read_text())
    records=[json.loads(line) for line in (directory/'pose.jsonl').read_text().splitlines()]
    for record in records:
        if record.get('valid'):
            record['xy']={name:np.asarray(xy) for name,xy in record['xy'].items()}
    fps=report['video']['fps'];results=[]
    qc_path=directory/'quality_review.json'
    qc=json.loads(qc_path.read_text()) if qc_path.exists() else {}
    for candidate in report['candidates']:
        start,end=candidate['shot_start'],candidate['shot_end']
        valid=[r for r in records if r.get('valid') and start<=r['index']<=end]
        measurements=_metrics(valid,None,None,fps)
        results.append({
            'proposal_id':candidate['shot_id'],
            'source_start_frame':start,'source_end_frame':end,
            'start_seconds':start/fps,'end_seconds':end/fps,
            'valid_frame_coverage':len(valid)/(end-start+1),
            'coach_approved':False,'outcome':'unknown',
            'tracking_review_failed':bool(qc.get('tracking_issue')),
            'metrics':None if qc.get('tracking_issue') else measurements.model_dump(mode='json'),
            'limitations':[
                qc.get('note_he','Unreviewed automatic window; not a confirmed shot.'),
                '2D projected measurements depend on camera view.',
                'Both visible body sides may contribute to pooled angle percentiles.',
                'Min/max angle fields represent 5th/95th percentiles in the existing engine.',
                'No release-specific measurements or causal coaching inferred here.',
            ],
        })
    (directory/'proposal_metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    return results


if __name__=='__main__':
    import sys
    root=Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/club_pilot')
    manifest=json.loads((root/'all_manifest.json').read_text())
    for video in manifest['videos']:
        directory=root/video['id']
        if (directory/'report.json').exists() and (directory/'pose.jsonl').exists():
            print(video['id'],len(export_metrics(directory)))
