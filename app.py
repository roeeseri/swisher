from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cv2
import streamlit as st
import streamlit.components.v1 as components

from swisher.session import SessionStore, ShotAnnotation, extract_clip, read_frame
from swisher.ui import report_view, theme


def analyze_video(*args, **kwargs):
    # Load expensive vision libraries only when the coach starts analysis.
    from swisher.analysis import analyze_video as run

    return run(*args, **kwargs)


ROOT = Path(__file__).resolve().parent
STORE = SessionStore(ROOT / "data" / "sessions")
recorder = components.declare_component("swisher_recorder", path=str(ROOT / "components/recorder"))
st.set_page_config(page_title="Swisher · Coach Studio", page_icon="🏀", layout="wide")
theme()
st.markdown(
    """<div class="hero"><div class="eyebrow">SWISHER / COACH STUDIO · PILOT 0.3</div>
<h1>מהמגרש. לתנועה. לדגש הבא.</h1>
<p>מצלמים שחקן, בודקים את הזריקה ביחד, ובוחרים דגש אחד לסט הבא.</p></div>""",
    unsafe_allow_html=True,
)
with st.sidebar:
    st.header("🏀 Swisher")
    st.caption("סביבת העבודה של המאמן")
    st.write("① צילום או העלאה\n\n② זיהוי ותצוגת גוף\n\n③ אישור הזריקות\n\n④ ניתוח ודגש מאמן")
    st.divider()
    st.info("פיילוט מקומי • ניתוח שחקן אחד בכל פעם • אין זיהוי זהות אוטומטי")
    st.caption("הקבצים נשמרים במחשב שלך, בתיקיית הפרויקט שעל שולחן העבודה.")

sessions = STORE.list_sessions()
capture_tab, workspace_tab, guide_tab = st.tabs(["אימון חדש", "סביבת ניתוח", "לקראת הצילום"])
with guide_tab:
    st.subheader("מגיעים למגרש מוכנים")
    st.write(
        "התחילו ב־3–5 זריקות, 20–40 שניות, שחקן אחד וכדור אחד בכל קליפ. "
        "הניחו טלפון על חצובה, בתאורה טובה; העדיפו 1080p ו־60 FPS."
    )
    st.write(
        "צלמו את כל הגוף והכדור. אם שחקנים אחרים ברקע, סמנו בסביבת הניתוח "
        "אזור שמכיל רק את הזורק. יש לשמור את המצלמה קבועה."
    )
    st.warning(
        "צילום טלפון: כרגע אפשר לצלם במצלמת הטלפון ולהעביר את הקובץ למחשב. "
        "צילום ישיר בדפדפן מיועד למכשיר שעליו האפליקציה פתוחה ב־localhost. "
        "טרם הוקמה אפליקציית מובייל או העלאה מרחוק מאובטחת."
    )
    st.write(
        "קבלו הרשאה לצילום ולשימוש בחומר, כולל טיפול מתאים בקטינים. "
        "המערכת אינה מנהלת עבורכם הסכמות משפטיות. השתמשו במזהים ללא שמות מלאים."
    )
    st.caption("אין כרגע דיוק מאומת על הקבוצה שלכם. מספר הבדיקות בתוכנה אינו מדד לדיוק קליעה.")

with capture_tab:
    st.subheader("① מתחילים בסשן קצר")
    c1, c2, c3 = st.columns(3)
    player_id = c1.text_input("מזהה שחקן", value="player_001")
    coach_id = c2.text_input("מזהה מאמן", value="coach_001")
    phases = {"baseline": "לפני דגש", "intervention": "תרגול הדגש", "post_intervention": "אחרי דגש"}
    phase = c3.selectbox("שלב באימון", list(phases), format_func=phases.get)
    consent = st.checkbox("יש לי הרשאה לצלם ולשמור את הסרטון לניתוח ולתיוג מקומי")
    source = st.radio("מקור הסרטון", ["העלאת סרטון", "צילום מהמצלמה"], horizontal=True)
    video_bytes, video_name = None, None
    if source == "העלאת סרטון":
        uploaded = st.file_uploader(
            "גררו לכאן את הסרטון", type=["mp4", "mov", "m4v", "avi", "webm"]
        )
        if uploaded:
            video_bytes, video_name = uploaded.getvalue(), uploaded.name
            st.caption(f"{uploaded.name} · {uploaded.size / 1024**2:.1f} MB")
    elif consent:
        recording = recorder(key="capture")
        if recording and recording.get("id") != st.session_state.get("saved_recording"):
            try:
                video_bytes = base64.b64decode(recording["data"], validate=True)
                video_name = recording["name"]
            except (KeyError, ValueError):
                st.error("העברת ההקלטה נכשלה. נסו להעלות את העותק שהורדתם.")
    else:
        st.info("אשרו הרשאה לשמירת הסרטון כדי לפתוח את מסך הצילום.")
    if st.button(
        "שמור סשן ופתח סביבת תיוג ←", type="primary", disabled=not (video_bytes and consent)
    ):
        try:
            new_id = datetime.now(UTC).strftime("session_%Y%m%d_%H%M%S_") + uuid4().hex[:4]
            STORE.create(
                session_id=new_id,
                player_id=player_id,
                coach_id=coach_id,
                phase=phase,
                camera_view="fixed",
                consent_confirmed=consent,
                video_name=video_name,
                video_bytes=video_bytes,
            )
            st.session_state["selected_session"] = new_id
            if source == "צילום מהמצלמה":
                st.session_state["saved_recording"] = recording["id"]
            st.session_state["created_notice"] = True
            st.rerun()
        except Exception as exc:
            st.error(f"הסשן לא נשמר: {exc}")
    if st.session_state.get("created_notice"):
        st.success("הסשן נשמר. עברו ללשונית ׳סביבת ניתוח׳ כדי לזהות את התנועה ולתייג.")

with workspace_tab:
    if not sessions:
        st.info("סביבת הניתוח תיפתח כאן אחרי שמירת הסשן הראשון.")
    else:
        preferred = st.session_state.get("selected_session", sessions[-1])
        session_id = st.selectbox(
            "הסשן שלך", sessions, index=sessions.index(preferred) if preferred in sessions else 0
        )
        metadata = STORE.load(session_id)
        directory = STORE.directory(session_id)
        source_path = STORE.video_path(session_id)
        annotations = STORE.annotations(session_id)
        cols = st.columns(4)
        cols[0].metric("שחקן", metadata.player_id)
        cols[1].metric("אורך הקליפ", f"{metadata.frame_count / metadata.fps:.1f} שנ׳")
        cols[2].metric("קצב צילום", f"{metadata.fps:.0f} FPS")
        cols[3].metric("זריקות מאושרות", len(annotations))
        st.subheader("② הגוף והתנועה")
        left, right = st.columns([1.5, 1])
        left.video(str(source_path))
        region = None
        with right:
            st.write("סריקה אחת יוצרת וידאו עם שלד והצעות לזריקות. המאמן מאשר או מתקן אותן.")
            st.caption("שתי דקות לכל היותר. העיבוד עשוי להימשך מספר דקות; הוא אינו בזמן אמת.")
            select_region = st.checkbox("יש עוד שחקנים? בחר אזור לזורק", key=f"roi_{session_id}")
            if select_region:
                horiz = st.slider("גבולות אופקיים (%)", 0, 100, (10, 90), key=f"x_{session_id}")
                vert = st.slider("גבולות אנכיים (%)", 0, 100, (0, 100), key=f"y_{session_id}")
                region = (
                    int(horiz[0] * metadata.width / 100),
                    int(vert[0] * metadata.height / 100),
                    int(horiz[1] * metadata.width / 100),
                    int(vert[1] * metadata.height / 100),
                )
                preview = read_frame(source_path, 0)
                cv2.rectangle(preview, region[:2], region[2:], (100, 240, 200), 3)
                st.image(preview, caption="כל הגוף והיד המורמת צריכים להישאר באזור הירוק")
            thorough = st.toggle("סריקה צפופה · זיהוי כדור בכל פריים", value=False)
            st.caption("צפופה = יותר דגימות ופחות דילוגי זמן; אינה מבטיחה דיוק גבוה יותר.")
            scan = st.button("זהה גוף והצע תיוגים", type="primary", use_container_width=True)
        pointer = directory / "latest_scan.json"
        if scan:
            run = directory / "runs" / uuid4().hex
            run.mkdir(parents=True)
            bar = st.progress(0, text="טוען מודלים מקומיים…")
            try:
                result = analyze_video(
                    source_path,
                    run / "overlay.mp4",
                    on_progress=lambda v: bar.progress(v, text=f"מזהה תנועה · {v:.0%}"),
                    player_region=region,
                    thorough=thorough,
                    trace_path=run / "pose.jsonl",
                )
                STORE._atomic_json(run / "report.json", result.model_dump(mode="json"))
                STORE._atomic_json(pointer, {"run_id": run.name})
                bar.progress(1.0, text="הסריקה הושלמה. עכשיו בודקים את ההצעות.")
            except Exception as exc:
                st.error(f"הסריקה לא הושלמה: {exc}. אפשר לתייג ידנית ולהמשיך.")
                st.caption("תוצאות מסריקות קודמות, אם קיימות, לא הוחלפו.")
        report = None
        if pointer.exists():
            run_id = json.loads(pointer.read_text())["run_id"]
            if not (len(run_id) == 32 and all(c in "0123456789abcdef" for c in run_id)):
                st.error("מזהה סריקה אינו תקין")
                st.stop()
            run = directory / "runs" / run_id
            report = json.loads((run / "report.json").read_text())
            st.video(str(run / "overlay.mp4"))
            coverage = report["pose_confidence"]
            st.caption(
                f"תוצאת הסריקה השמורה · כיסוי גוף {coverage:.0%} · "
                f"{len(report['candidates'])} הצעות לבדיקה · ההגדרות שמורות בדוח"
            )
            if coverage < 0.55:
                st.warning(
                    "כיסוי הגוף נמוך. בדקו את אזור הזורק ואת התאורה; אין להסיק מכאן דגש טכני."
                )
            if not report["candidates"]:
                st.info("לא נמצאו הצעות בטוחות מספיק. אפשר עדיין לסמן זריקות ידנית למטה.")
        st.subheader("③ מאשרים את מה שבאמת קרה")
        proposals = report["candidates"] if report else []
        choices = {"manual": "＋ הוספת זריקה ידנית"}
        for p in proposals:
            choices[f"auto:{p['shot_id']}"] = f"הצעה {p['shot_id']} · {p['evidence']}"
        for a in annotations:
            choices[f"edit:{a.shot_id}"] = f"עריכת {a.shot_id} · גרסה {a.annotation_version}"
        choice = st.selectbox("הצעה לבדיקה או זריקה לעריכה", list(choices), format_func=choices.get)
        initial = {}
        if choice.startswith("auto:"):
            initial = next(p for p in proposals if p["shot_id"] == choice[5:])
        if choice.startswith("edit:"):
            initial = next(a.model_dump() for a in annotations if a.shot_id == choice[5:])
        k = f"{session_id}_{choice}"
        focus_frame = initial.get("release_frame") or initial.get("motion_frame", 0)
        frame = st.slider(
            "בדיקה פריים־פריים", 0, metadata.frame_count - 1, int(focus_frame), key=f"frame_{k}"
        )
        c1, c2 = st.columns([1.2, 1])
        c1.image(
            read_frame(source_path, frame),
            caption=f"פריים {frame} · כ־{frame / metadata.fps:.3f} שנ׳",
        )
        with c2:
            st.caption("בדקו את הפריימים בסליידר והעתיקו את המספרים לשדות המתאימים.")
            with st.form(f"annotation_{k}"):
                shot_id = st.text_input(
                    "מזהה זריקה", initial.get("shot_id", f"manual_{len(annotations) + 1:03d}")
                )
                start = st.number_input(
                    "תחילת זריקה (פריים)",
                    0,
                    metadata.frame_count - 1,
                    int(initial.get("shot_start", 0)),
                )
                known_release = st.checkbox(
                    "רגע השחרור נראה ואומת",
                    value=choice.startswith("edit:") and initial.get("release_frame") is not None,
                )
                release = st.number_input(
                    "פריים שחרור",
                    0,
                    metadata.frame_count - 1,
                    int(initial.get("release_frame") or frame),
                )
                end = st.number_input(
                    "סוף זריקה (פריים)",
                    0,
                    metadata.frame_count - 1,
                    int(initial.get("shot_end", metadata.frame_count - 1)),
                )
                outcomes = ["unknown", "made", "missed"]
                outcome = st.selectbox(
                    "תוצאת הזריקה",
                    outcomes,
                    index=outcomes.index(initial.get("outcome", "unknown")),
                    format_func=lambda x: {
                        "unknown": "לא ניתן לקבוע",
                        "made": "נכנסה",
                        "missed": "החטאה",
                    }[x],
                )
                evidence = st.text_area("מה ראית כמאמן?", initial.get("coach_evidence", ""))
                cue = st.text_input("דגש אחד לשחקן", initial.get("coach_cue", ""))
                drill = st.text_input("תרגיל לסט הבא", initial.get("drill", ""))
                approved = st.checkbox(
                    "הדגש נבדק ומאושר על ידי המאמן", value=initial.get("coach_approved", False)
                )
                save = st.form_submit_button("אישור ושמירת הזריקה", type="primary")
                if save:
                    try:
                        STORE.upsert_annotation(
                            session_id,
                            ShotAnnotation(
                                shot_id=shot_id,
                                shot_start=int(start),
                                release_frame=int(release) if known_release else None,
                                shot_end=int(end),
                                outcome=outcome,
                                coach_evidence=evidence,
                                coach_cue=cue,
                                drill=drill,
                                coach_approved=approved,
                                origin="auto_reviewed"
                                if choice.startswith("auto:")
                                else initial.get("origin", "manual"),
                            ),
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"התיוג לא נשמר: {exc}")
        st.subheader("④ קליפ, ראיות ודגש לסט הבא")
        if annotations:
            selected = st.selectbox("זריקה מאושרת לניתוח", [a.shot_id for a in annotations])
            annotation = next(a for a in annotations if a.shot_id == selected)
            artifact = (
                STORE.artifact_directory(session_id, selected) / f"v{annotation.annotation_version}"
            )
            result_path = artifact / "report.json"
            result_pointer = artifact / "latest.json"
            if result_pointer.exists():
                last_id = json.loads(result_pointer.read_text())["run_id"]
                if len(last_id) != 32 or any(c not in "0123456789abcdef" for c in last_id):
                    st.error("מזהה תוצאת זריקה אינו תקין")
                    st.stop()
                result_path = artifact / last_id / "report.json"
            if st.button("נתח את הזריקה המאושרת", type="primary"):
                run_dir = artifact / uuid4().hex
                run_dir.mkdir(parents=True, exist_ok=True)
                bar = st.progress(0, text="מכין קליפ לניתוח…")
                try:
                    extract_clip(
                        source_path,
                        run_dir / "clip.mp4",
                        annotation.shot_start,
                        annotation.shot_end,
                    )
                    result = analyze_video(
                        run_dir / "clip.mp4",
                        run_dir / "overlay.mp4",
                        on_progress=lambda v: bar.progress(v, text="בודק את הזריקה…"),
                        player_region=region,
                        thorough=True,
                        trace_path=run_dir / "pose.jsonl",
                    )
                    result.provenance.update(
                        {
                            "session_id": session_id,
                            "shot_id": selected,
                            "annotation_version": annotation.annotation_version,
                            "source_frame_offset": annotation.shot_start,
                            "confirmed_release_source_frame": annotation.release_frame,
                            "coach_id": metadata.coach_id,
                        }
                    )
                    result_path = run_dir / "report.json"
                    STORE._atomic_json(result_path, result.model_dump(mode="json"))
                    STORE._atomic_json(result_pointer, {"run_id": run_dir.name})
                    bar.progress(1.0)
                except Exception as exc:
                    st.error(f"הניתוח לא הושלם: {exc}")
            if annotation.coach_approved and annotation.coach_cue:
                st.success(f"הדגש שאישר המאמן: {annotation.coach_cue}")
                st.write(annotation.drill)
                st.caption(f"הראיה של המאמן: {annotation.coach_evidence}")
            if result_path.exists():
                shot_report = json.loads(result_path.read_text())
                st.video(str(result_path.parent / "overlay.mp4"))
                report_view(shot_report)
                if annotation.release_frame is not None:
                    predicted = shot_report.get("release")
                    if predicted:
                        delta = abs(
                            predicted["frame"] + annotation.shot_start - annotation.release_frame
                        )
                        st.caption(
                            f"פער בין הערכת המודל לתיוג המאמן: {delta} פריימים. בדיקה לזריקה זו בלבד."
                        )
                st.download_button(
                    "הורדת דוח הזריקה",
                    result_path.read_bytes(),
                    file_name=f"{selected}_v{annotation.annotation_version}.json",
                )
            st.download_button(
                "הורדת כל התיוגים",
                json.dumps(
                    [a.model_dump(mode="json") for a in annotations], ensure_ascii=False, indent=2
                ),
                file_name=f"{session_id}_annotations.json",
            )
        else:
            st.info("אשרו לפחות זריקה אחת כדי לקבל קליפ אישי וניתוח מפורט.")
