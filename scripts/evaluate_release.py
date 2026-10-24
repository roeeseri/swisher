"""Compare release proposals against explicitly provisional visual reference intervals."""
from __future__ import annotations

import json
from pathlib import Path


def evaluate(report: dict, reference: dict) -> dict:
    if report.get('provenance',{}).get('source_sha256') != reference.get('source_sha256'):
        raise ValueError('reference labels and video report have different source hashes')
    positives=[s for s in reference['shots'] if s.get('is_shot',True)]
    negatives=[s for s in reference['shots'] if not s.get('is_shot',True)]
    predictions=[p for p in report['candidates'] if p.get('release_frame') is not None]
    matched=set();events=[]
    for truth in positives:
        lo,hi=truth['release_interval']
        options=[(i,p) for i,p in enumerate(predictions) if i not in matched
                 and (p.get('release_interval') or [p['release_frame']]*2)[0]<=hi
                 and (p.get('release_interval') or [p['release_frame']]*2)[1]>=lo]
        if options:
            i,p=min(options,key=lambda item:abs(item[1]['release_frame']-(lo+hi)/2))
            matched.add(i)
            events.append({'reference_interval':[lo,hi],'predicted_frame':p['release_frame'],
                           'predicted_interval':p.get('release_interval'),'matched':True})
        else:events.append({'reference_interval':[lo,hi],'matched':False})
    false_in_reviewed_negatives=[p['release_frame'] for p in predictions
        if any(n['window'][0]<=p['release_frame']<=n['window'][1] for n in negatives)]
    return {'reference_shots':len(positives),'matched_release_intervals':len(matched),
            'missed_reference_releases':len(positives)-len(matched),
            'predicted_releases':len(predictions),
            'false_releases_in_reviewed_negative_windows':false_in_reviewed_negatives,
            'unmatched_predictions_requiring_review':len(predictions)-len(matched),
            'events':events}


def main():
    root=Path('artifacts/club_pilot/updated');rows=[]
    for name in ['IMG_1083','IMG_1078','IMG_1068']:
        current=root/name/'report.json'
        if not current.exists():continue
        ref=json.loads(Path(f'artifacts/release_study/{name}_reference.json').read_text())
        old=json.loads(Path(f'artifacts/club_pilot/{name}/report.json').read_text())
        new=json.loads(current.read_text())
        rows.append({'video':name,'role':'development' if name=='IMG_1083' else 'holdout',
                     'baseline':evaluate(old,ref),'updated':evaluate(new,ref)})
    out=root/'evaluation';out.mkdir(exist_ok=True)
    payload={'reference_status':'Provisional, assistant visual review guided by old proposals. Not exhaustive full-video labels and not coach approved. No population accuracy claim.',
             'matching':'One-to-one overlap between predicted interval and visual reference interval. Predictions outside reviewed windows remain unverified.',
             'results':rows}
    (out/'comparison.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    table=''.join(f"<tr><td>{r['video']}</td><td>{'פיתוח' if r['role']=='development' else 'בדיקה'}</td><td>{r['baseline']['reference_shots']}</td><td>{r['baseline']['matched_release_intervals']}</td><td>{r['updated']['matched_release_intervals']}</td><td>{len(r['updated']['false_releases_in_reviewed_negative_windows'])}</td></tr>" for r in rows)
    text='''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><title>השוואת זיהוי שחרור</title><style>body{font:18px Arial;line-height:1.7;max-width:1000px;margin:40px auto;padding:20px;color:#172331}table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:right;border-bottom:1px solid #ddd}</style><h1>בדיקת השיפור בזיהוי השחרור</h1><p>ההשוואה נעשתה מול חלונות שחרור שסומנו בבדיקה חזותית של פריימים. אלה סימוני ייחוס זמניים, בהכוונת הצעות הגרסה הישנה; הם לא תיוג מלא ועצמאי של כל הסרטונים ולא אישור מאמן.</p><p>התאמה נספרת כאשר חלון ההפרדה המשוער של המודל חופף לחלון הייחוס, בהתאמה של אחד לאחד. מספר הסימונים החדשים אינו מדד לדיוק.</p><table><tr><th>סרטון</th><th>תפקיד</th><th>חלונות ייחוס</th><th>התאמות לפני</th><th>התאמות אחרי</th><th>סימוני שחרור בחלונות שליליים שנבדקו</th></tr>'''+table+'''</table><p>הבדיקה עדיין מצומצמת: יש החמצות, זיהוי שחקנית לא אומת בכל פריים, ותוצאות קליעה והחטאה לא תויגו. אין להסיק ממנה שהמערכת מאבחנת טכניקה או מסבירה סיבת החטאה.</p><p><a href="comparison.json">נתוני ההשוואה</a> · <a href="../results.html">תוצאות כל הסרטונים</a></p></html>'''
    (out/'index.html').write_text(text,encoding='utf-8');print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
