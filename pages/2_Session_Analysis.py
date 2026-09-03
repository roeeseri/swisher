from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from swisher.analysis import analyze_video
from swisher.session import SessionStore, extract_clip

ROOT = Path(__file__).resolve().parents[1]
STORE = SessionStore(ROOT / "data" / "sessions")

st.set_page_config(page_title="Swisher Session Analysis", page_icon="🧠", layout="wide")
st.markdown("<style>.stApp { direction: rtl; }</style>", unsafe_allow_html=True)
st.title("🧠 ניתוח סשן")
st.caption("כל זריקה שסומנה נחתכת, מנותחת ונשמרת עם וידאו מסומן ודוח JSON.")

sessions = STORE.list_sessions()
if not sessions:
    st.info("אין עדיין סשנים. צור ותייג סשן בעמוד Swisher Session Studio.")
    st.stop()

session_id = st.selectbox("בחר סשן", sessions, index=len(sessions) - 1)
metadata = STORE.load(session_id)
annotations = STORE.annotations(session_id)
if not annotations:
    st.warning("הסשן עדיין לא כולל זריקות מסומנות.")
    st.stop()

shot_ids = [item.shot_id for item in annotations]
selected_shots = st.multiselect(
    "בחר זריקות לניתוח", shot_ids, default=shot_ids[: min(3, len(shot_ids))]
)
st.warning(
    "הניתוח המקומי משתמש במודלי Vision כבדים. בפעם הראשונה הוא עשוי להימשך "
    "מספר דקות לכל זריקה. מומלץ להתחיל ב־1–3 זריקות."
)

if st.button(
    "נתח את הזריקות שנבחרו",
    type="primary",
    use_container_width=True,
    disabled=not selected_shots,
):
    overall = st.progress(0, text="מתחיל…")
    source = STORE.video_path(session_id)
    selected = [item for item in annotations if item.shot_id in selected_shots]
    for index, annotation in enumerate(selected):
        st.write(f"**{annotation.shot_id}**")
        artifact_dir = STORE.artifact_directory(session_id, annotation.shot_id)
        clip_path = artifact_dir / "source_clip.mp4"
        annotated_path = artifact_dir / "annotated.mp4"
        report_path = artifact_dir / "report.json"
        try:
            extract_clip(
                source,
                clip_path,
                max(0, annotation.shot_start),
                min(metadata.frame_count - 1, annotation.shot_end),
            )

            def progress(
                value: float,
                current_index: int = index,
                current_shot_id: str = annotation.shot_id,
            ) -> None:
                total = (current_index + value) / len(selected)
                overall.progress(total, text=f"מנתח {current_shot_id}…")

            result = analyze_video(clip_path, annotated_path, on_progress=progress)
            report_path.write_text(
                json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if result.status == "ok":
                st.success(f"{annotation.shot_id}: הניתוח הושלם")
            else:
                st.warning(f"{annotation.shot_id}: {result.message}")
        except Exception as exc:
            st.error(f"{annotation.shot_id}: הניתוח נכשל — {exc}")
    overall.progress(1.0, text="הסשן הסתיים")

st.divider()
st.subheader("תוצאות קיימות")
for annotation in annotations:
    artifact_dir = STORE.artifact_directory(session_id, annotation.shot_id)
    report_path = artifact_dir / "report.json"
    annotated_path = artifact_dir / "annotated.mp4"
    if not report_path.exists():
        continue
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with st.expander(f"{annotation.shot_id} · {report['status']}"):
        result_cols = st.columns(4)
        result_cols[0].metric("Pose", f"{report['pose_confidence']:.0%}")
        result_cols[1].metric("זיהוי כדור", f"{report['detections']['ball_detection_rate']:.0%}")
        release = report.get("release")
        result_cols[2].metric("שחרור", f"{release['seconds']:.2f}s" if release else "לא זוהה")
        outcome = report.get("outcome", {}).get("status", "unknown")
        result_cols[3].metric("תוצאה", outcome)
        coaching = report.get("coaching", {})
        st.write(f"**דגש:** {coaching.get('observation', 'לא זמין')}")
        if annotated_path.exists():
            st.video(str(annotated_path))
        st.download_button(
            "הורד דוח JSON",
            report_path.read_bytes(),
            file_name=f"{session_id}_{annotation.shot_id}.json",
            mime="application/json",
            key=f"download_{annotation.shot_id}",
        )
