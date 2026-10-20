"""Build a private, self-contained coach review ZIP. No server or Python required."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cv2

OFFLINE = r"""
const COACH_PACKAGE=__PACKAGE__;
const packageKey='swisher-coach-'+COACH_PACKAGE.package_id;
const memoryStore=new Map();let durable=true;
const coachStorage={getItem(k){try{return localStorage.getItem(k)}catch(e){return memoryStore.get(k)||null}},setItem(k,v){try{localStorage.setItem(k,v)}catch(e){durable=false;memoryStore.set(k,v)}}};
let coachData=JSON.parse(JSON.stringify(COACH_PACKAGE.state));
try{const saved=coachStorage.getItem(packageKey);if(saved)coachData=JSON.parse(saved)}catch(e){}
function persistCoach(){coachStorage.setItem(packageKey,JSON.stringify(coachData));if(!durable){const n=document.getElementById('offlineNotice');if(n)n.textContent='הדפדפן אינו מאפשר שמירה מקומית. ייצאו את הסקירה לפני סגירה.'}}
function coachExport(){return {schema_version:1,package_id:COACH_PACKAGE.package_id,exported_at:new Date().toISOString(),jobs:coachData.jobs,shots:coachData.shots.map(s=>({shot_id:s.id,job_id:s.job_id,candidate:s.candidate,suggestion:s.suggestion,base_version:COACH_PACKAGE.base_versions[s.id]||0,reviews:s.reviews}))}}
window.fetch=async(url,options={})=>{
 const path=String(url).split('?')[0];
 try{
  if(path==='/api/state')return new Response(JSON.stringify(coachData),{status:200});
  if(path==='/api/review'){
   const p=JSON.parse(options.body),s=coachData.shots.find(s=>s.id===p.id);if(!s)throw Error('הזריקה לא נמצאה');
   if(p.version!==s.reviews.length)throw Error('הפריט השתנה. רעננו ונסו שוב');
   if(!['approved','corrected','uncertain','not_shot'].includes(p.decision))throw Error('בחרו החלטה');
   if(!p.coach?.trim()||p.coach.length>80)throw Error('יש למלא מזהה מאמן');
   if(p.decision==='approved'&&!s.suggestion.available)throw Error('אין הערה מוצעת לאישור');
   if(p.decision==='corrected'&&!p.note?.trim())throw Error('יש לכתוב הערה חלופית');
   if((p.note||'').length>3000)throw Error('הערה ארוכה מדי');
   const j=coachData.jobs.find(j=>j.id===s.job_id);
   if(!Number.isInteger(p.start_frame)||!Number.isInteger(p.end_frame)||p.start_frame<0||p.start_frame>=p.end_frame||p.end_frame>=j.video.frame_count)throw Error('טווח לא תקין');
   if(p.release_frame!==null&&(!Number.isInteger(p.release_frame)||p.release_frame<p.start_frame||p.release_frame>p.end_frame))throw Error('פריים שחרור לא תקין');
   const r={...p,version:s.reviews.length+1,review_id:crypto.randomUUID?.()||Date.now()+'-'+Math.random(),created_at:new Date().toISOString(),note:p.decision==='approved'?s.suggestion.text:(p.note||'').trim()};
   delete r.id;s.reviews.push(r);persistCoach();return new Response(JSON.stringify(r),{status:200});
  }
  if(path==='/api/job-label'){
   const p=JSON.parse(options.body),j=coachData.jobs.find(j=>j.id===p.id);if(!j||!p.player?.trim())throw Error('יש למלא מזהה שחקן');j.player=p.player.trim();persistCoach();return new Response('{"ok":true}',{status:200});
  }
  throw Error('הפעולה הזו זמינה במערכת המלאה במחשב של רואי');
 }catch(e){return new Response(JSON.stringify({error:e.message}),{status:400})}
};
document.addEventListener('DOMContentLoaded',()=>{
 const a=document.querySelector('a.export-button');a.textContent='סיום וייצוא הסקירה ↗';
 a.onclick=e=>{e.preventDefault();const blob=new Blob([JSON.stringify(coachExport(),null,2)],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='Swisher_feedback_'+(coachStorage.getItem('swisherCoach')||'coach').replace(/[^a-zA-Z0-9_-]/g,'_')+'_'+new Date().toISOString().slice(0,10)+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),10000);};
 const note=document.createElement('div');note.id='offlineNotice';note.className='offline-instructions';note.textContent='העבודה נשמרת בדפדפן הזה. בסיום לחצו ״סיום וייצוא הסקירה״ ושלחו לרואי את קובץ התיוגים שהורד.';document.querySelector('#page-review .hero').after(note);
 if(location.hash==='#capture')location.hash='#review';
});
"""


def build(root: Path, destination: Path):
    raw = json.loads((root / "workspace.json").read_text())
    if any(s["media_status"] != "ready" for s in raw["shots"]):
        raise ValueError("Wait until all shot clips are ready before export")
    destination.mkdir(parents=True, exist_ok=True)
    package_id = uuid4().hex
    jobs = [
        {
            key: j.get(key)
            for key in (
                "id",
                "name",
                "player",
                "view",
                "status",
                "video",
                "source_sha256",
                "capture_id",
                "segment",
                "progress",
                "error",
            )
        }
        for j in raw["jobs"]
    ]
    shots = json.loads(json.dumps(raw["shots"]))
    regions = {}
    for job in raw["jobs"]:
        report = job.get("report") or job.get("existing_report")
        lookup = {}
        if report and (Path(report).parent / "pose.jsonl").exists():
            for line in (Path(report).parent / "pose.jsonl").open():
                row = json.loads(line)
                if row.get("camera_tracking_valid", True) and row.get("player_region"):
                    lookup[row["index"]] = row["player_region"]
        regions[job["id"]] = lookup
    for shot in shots:
        folder = destination / "media" / shot["id"]
        folder.mkdir(parents=True, exist_ok=True)
        source = root / "shots" / shot["id"]
        shutil.copy2(source / "clip.mp4", folder / "clip.mp4")
        image = cv2.imread(str(source / "frame.jpg"))
        if image is None:
            raise ValueError(f"Missing source frame: {shot['id']}")
        c = shot["candidate"]
        latest = shot["reviews"][-1] if shot["reviews"] else None
        index = latest["release_frame"] if latest else c.get("release_frame")
        if index is None:
            index = c.get("motion_frame", c["shot_start"])
        region = regions[shot["job_id"]].get(index)
        job = next(j for j in jobs if j["id"] == shot["job_id"])
        if region:
            sx = image.shape[1] / job["video"]["width"]
            sy = image.shape[0] / job["video"]["height"]
            x1, y1, x2, y2 = [round(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(region)]
            crop = image[max(0, y1) : min(image.shape[0], y2), max(0, x1) : min(image.shape[1], x2)]
            if crop.size:
                image = crop
        if not cv2.imwrite(str(folder / "frame.jpg"), image):
            raise ValueError("Frame export failed")
    state = {"version": 1, "jobs": jobs, "shots": shots}
    package = {
        "package_id": package_id,
        "created_at": datetime.now(UTC).isoformat(),
        "state": state,
        "base_versions": {s["id"]: len(s["reviews"]) for s in shots},
    }
    html = Path("src/swisher/studio/index.html").read_text()
    adapter = OFFLINE.replace(
        "__PACKAGE__", json.dumps(package, ensure_ascii=False).replace("</", "<\\/")
    )
    html = html.replace("const TOKEN='__TOKEN__'", "const TOKEN='offline'")
    html = html.replace("localStorage.getItem", "coachStorage.getItem").replace(
        "localStorage.setItem", "coachStorage.setItem"
    )
    html = html.replace("<script>", "<script>" + adapter + "</script>\n<script>", 1)
    html = html.replace("/media/${s.id}/clip.mp4", "media/${s.id}/clip.mp4")
    html = html.replace("/media/${s.id}/preview.jpg?frame=${frame}", "media/${s.id}/frame.jpg")
    # A pack contains the clip and the selected still, not all frames of the raw source.
    # Frame/bounds correction remains in the full application, never simulated offline.
    html = html.replace(
        "</style>",
        """
    #page-capture,nav [data-page=capture],#phoneConnect,#addManual,.advanced,.frame-scrub{display:none!important}
    .offline-instructions{padding:12px 16px;border:1px solid #34475d;border-radius:10px;margin-bottom:16px;color:#b9c9de;font-size:13px;background:#152235}
    </style>""",
    )
    html = html.replace(
        "$('connectionState').textContent='● מחובר'",
        "$('connectionState').textContent='● עבודה מקומית'",
    )
    (destination / "index.html").write_text(html)
    (destination / "package_manifest.json").write_text(
        json.dumps(
            {"package_id": package_id, "jobs": jobs, "base_versions": package["base_versions"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    (destination / "README_HE.txt").write_text("""Swisher — חבילת עבודה למאמן / מאמנת

1. חלצו את קובץ ה־ZIP במלואו לתיקייה. אל תפתחו מתוך ה־ZIP.
2. פתחו את index.html בדפדפן במחשב. אין צורך בהתקנת Python או בהרצת שרת.
3. לחצו על "בחירת מאמן" והזינו מזהה קבוע שלכם.
4. לכל זריקה: צפו בסרטון ובתמונת השחרור, ואשרו את ההערה או כתבו אחרת.
   אפשר גם לסמן "לא ניתן לקבוע" או "זו לא זריקה". תוצאת קליעה היא שדה רשות.
5. לחצו על "שמירה והמשך". אפשר לעצור ולחזור באותו דפדפן ובאותו מחשב.
6. בסיום לחצו על "סיום וייצוא הסקירה". שלחו לרואי את קובץ ה־JSON שהורד.
   לא צריך להחזיר את כל הסרטונים. כדאי לייצא גם בסיום כל מקטע עבודה.

הערות המערכת הן הצעות ניסיוניות לבדיקה, ואינן אבחון מקצועי מאומת.
כאשר לא זוהה שחרור, התמונה מסומנת כפריים לבדיקה.
בחבילה זו ניתן לתייג הערות ותוצאות. תיקוני חיתוך ופריימים נעשים במערכת המלאה.

החבילה כוללת סרטוני קטינות: מיועדת למאמנים שהורשו לעבוד על פרויקט Swisher בלבד.
אין לפרסם או להעביר הלאה. שמרו אותה בתיקייה פרטית ומחקו בסיום התהליך לפי הנחיות רואי.
""")
    registry_path = root / "coach_packages.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    registry[package_id] = {
        "base_versions": package["base_versions"],
        "created_at": package["created_at"],
        "job_hashes": {j["id"]: j["source_sha256"] for j in jobs},
        "candidates": {
            s["id"]: hashlib.sha256(json.dumps(s["candidate"], sort_keys=True).encode()).hexdigest()
            for s in shots
        },
    }
    from swisher.session import SessionStore

    SessionStore._atomic_json(registry_path, registry)
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as out:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                out.write(path, Path(destination.name) / path.relative_to(destination))
    return archive, len(shots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/studio"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive, count = build(args.root, args.output)
    print(
        json.dumps(
            {"archive": str(archive.resolve()), "shots": count, "bytes": archive.stat().st_size}
        )
    )
