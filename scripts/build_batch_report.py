"""Build a local Hebrew results page from actual Swisher batch reports."""
from __future__ import annotations
import csv
import html
import json
import sys
from pathlib import Path


def build_report(root: Path) -> list[dict]:
    manifest=json.loads((root/'all_manifest.json').read_text())
    rows=[];cards=[];table=[]
    for item in manifest['videos']:
        name=item['id'];path=root/name/'report.json'
        if not path.exists():
            status='שגיאה' if (root/name/'error.txt').exists() else 'ממתין לעיבוד'
            table.append(f'<tr><td>{name}</td><td>{status}</td><td>—</td><td>—</td><td>—</td></tr>')
            continue
        try:r=json.loads(path.read_text())
        except json.JSONDecodeError:continue
        fps=r['video']['fps'];cand=r['candidates'];cov=r['pose_confidence'];releases=sum(c['release_frame'] is not None for c in cand)
        video_file='events.mp4' if (root/name/'events.mp4').exists() else 'overlay.mp4'
        rows.append({'video':name,'pose_coverage':round(cov,4),'proposals':len(cand),'proposed_releases':releases,'reviewed_shots':0,'duration_seconds':r['video']['duration_seconds'],'status':r['status']})
        status='נמנע ממסקנה' if r['status']=='abstained' else 'עבר סף ניתוח'
        qc_path=root/name/'quality_review.json'
        qc=json.loads(qc_path.read_text()) if qc_path.exists() else {}
        if qc.get('tracking_issue'):
            status='מעקב לא אמין בבדיקה'
            rows[-1]['status']='tracking_review_failed'
        table.append(f'<tr><td><a href="#{name}">{name}</a></td><td>{status}</td><td>{cov:.1%}</td><td>{len(cand)}</td><td>{releases}</td></tr>')
        trs=''.join(f"""<tr><td>{html.escape(c['shot_id'])}</td><td><button onclick="this.closest('section').querySelector('video').currentTime={c['shot_start']/fps:.3f};this.closest('section').querySelector('video').play()">{c['shot_start']/fps:.2f}</button></td><td>{c['shot_end']/fps:.2f}</td><td>{'לא זוהה' if c['release_frame'] is None else format(c['release_frame']/fps,'.2f')}</td><td>טרם אומתה</td></tr>""" for c in cand)
        warnings='; '.join(r['quality']['warnings']) or 'אין התרעת איכות כללית; עדיין נדרשת בדיקה חזותית.'
        if qc.get('note_he'):
            warnings=qc['note_he']+' '+warnings
        metrics_html = ''
        metrics_path=root/name/'proposal_metrics.json'
        if metrics_path.exists():
            per_proposal=json.loads(metrics_path.read_text())
            metric_rows=[]
            def interval(metrics, joint):
                lo=metrics.get('min_'+joint+'_angle_deg')
                hi=metrics.get('max_'+joint+'_angle_deg')
                return '—' if lo is None or hi is None else f'{lo:.0f}°–{hi:.0f}°'
            for entry in per_proposal:
                metrics=entry.get('metrics')
                if not metrics:
                    continue
                trunk=metrics.get('trunk_lean_range_deg')
                trunk_text='—' if trunk is None else f'{trunk:.1f}°'
                metric_rows.append(f"<tr><td>{entry['proposal_id']}</td><td>{entry['valid_frame_coverage']:.0%}</td><td dir='ltr'>{interval(metrics,'knee')}</td><td dir='ltr'>{interval(metrics,'elbow')}</td><td>{trunk_text}</td></tr>")
            if metric_rows:
                metrics_html='<details><summary>מדדי תנועה ניסיוניים לכל הצעה</summary><p>זוויות בהיטל התמונה, תלויות בזווית הצילום. הטווחים מייצגים אחוזונים 5 עד 95, ועשויים לשלב את שני צדי הגוף הנראים. אלה מדידות ראשוניות ולא ציון לטכניקה או הסבר להחטאה.</p><table><tr><th>הצעה</th><th>כיסוי גוף בחלון</th><th>טווח ברך</th><th>טווח מרפק</th><th>שינוי נטיית גו</th></tr>'+''.join(metric_rows)+'</table></details>'
        cards.append(f'''<section id="{name}"><h2>{name}</h2><p>כיסוי גוף: <b>{cov:.1%}</b> · <b>{len(cand)}</b> הצעות לזריקות · <b>{releases}</b> הצעות עם שחרור · זריקות מאומתות: <b>0</b></p><p>{html.escape(warnings)}</p><video controls preload="none" src="{name}/{video_file}"></video><p>השלד צריך לעקוב אחר הזורקת. כל ההצעות, השחרורים ותוצאות הזריקות טעונים אימות.</p><details><summary>זמני ההצעות</summary><table><tr><th>הצעה</th><th>התחלה בשניות</th><th>סיום בשניות</th><th>שחרור משוער בשניות</th><th>תוצאה</th></tr>{trs}</table></details>{metrics_html}<p><a href="{name}/report.json">דוח המודל המלא</a> · <a href="{name}/overlay.mp4">הסרטון המסומן</a> · <a href="{name}/proposal_metrics.json">מדדי תנועה לכל הצעה</a></p></section>''')
    if rows:
        with (root/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    count=len(rows);total=len(manifest['videos'])
    proposal_count=sum(row['proposals'] for row in rows)
    release_count=sum(row['proposed_releases'] for row in rows)
    tracking_failures=sum(row['status']=='tracking_review_failed' for row in rows)
    (root/'batch_findings.json').write_text(json.dumps({'completed':count,'total':total,'raw_proposals':proposal_count,'proposed_releases':release_count,'reviewed_shots':0,'known_tracking_failures':tracking_failures},ensure_ascii=False,indent=2))
    document='''<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Swisher — תוצאות הרצה</title><style>html{scroll-behavior:smooth}body{font:17px Arial;line-height:1.6;max-width:1080px;margin:40px auto;padding:0 24px;color:#172331;background:#f5f7fa}section{background:white;padding:25px;margin:25px 0;border:1px solid #e1e5eb;border-radius:12px}video{width:100%;background:#111}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}a{color:#215db0}summary{cursor:pointer;font-weight:bold}button{padding:10px 16px;cursor:pointer}h1{margin-bottom:0}</style><h1>Swisher — תוצאות סרטוני עוצמה</h1>'''
    document+=f'<p>הושלמו <b>{count} מתוך {total}</b> סרטונים. <button onclick="location.reload()">רענון תוצאות</button></p>'
    document+=f'<p><b>{proposal_count}</b> הצעות גולמיות לזריקות · <b>{release_count}</b> הצעות עם שחרור משוער · <b>{tracking_failures}</b> סרטונים שסומנו עם בעיית מעקב בבדיקה. תוצאות קליעה והחטאה לא אומתו.</p>'
    document+='''<p>פלט של מנוע Swisher 0.3. הצעות אינן ספירת זריקות מאומתת. כיסוי גוף הוא שיעור הפריימים שעברו את סף המערכת, כולל הזמן שבין הזריקות; הוא אינו אחוז דיוק. אין עדיין מסקנה מאומתת על סיבת החטאה או אחוזי קליעה.</p><p><a href="summary.csv">הורדת טבלת הסיכום</a></p><table><tr><th>סרטון</th><th>מצב המודל</th><th>כיסוי גוף</th><th>הצעות זריקה</th><th>שחרורים מוצעים</th></tr>'''+''.join(table)+'</table>'+''.join(cards)+'</html>'
    temp=root/'results.tmp.html';temp.write_text(document,encoding='utf-8');temp.replace(root/'results.html')
    return rows


if __name__=='__main__':
    root=Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/club_pilot')
    print(json.dumps(build_report(root),ensure_ascii=False,indent=2))
