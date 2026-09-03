from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from swisher.analysis import analyze_video


def _fmt(value: float | None, unit: str, scale: float = 1.0) -> str:
    if value is None:
        return "לא זמין"
    return f"{value * scale:.1f}{unit}"


st.set_page_config(page_title="Swisher Lab", page_icon="🏀", layout="wide")
st.markdown(
    """
    <style>
    .stApp { direction: rtl; }
    [data-testid="stFileUploader"] { direction: rtl; }
    .metric-note { color: #737373; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏀 Swisher Lab")
st.caption("מודל ניתוח זריקה — אב־טיפוס עובד v0.2")

with st.expander("מה המודל יודע כרגע?", expanded=False):
    st.write(
        "המודל מזהה שחקן וכדור עם RF-DETR, מריץ Pose מוגדל על השחקן, מאתר "
        "את חלון התנועה ומחשב מדדי מרפק, ברך ויציבות. סל ו-make/miss עדיין אינם פעילים."
    )

uploaded = st.file_uploader("העלה סרטון זריקה", type=["mp4", "mov", "m4v", "avi"])

if uploaded is None:
    st.info("מומלץ: סרטון קצר, מצלמה יציבה, כל הגוף בפריים ותאורה טובה.")
    st.stop()

st.video(uploaded.getvalue())

if st.button("נתח את הסרטון", type="primary", use_container_width=True):
    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    with tempfile.TemporaryDirectory(prefix="swisher_") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        output_path = Path(temp_dir) / "annotated.mp4"
        input_path.write_bytes(uploaded.getvalue())

        progress = st.progress(0, text="בודק את הסרטון ומפעיל את מודל הגוף…")

        def on_progress(value: float) -> None:
            progress.progress(min(int(value * 100), 100), text="מנתח את התנועה…")

        try:
            result = analyze_video(input_path, output_path, on_progress=on_progress)
        except Exception as exc:  # UI boundary: expose a useful message instead of crashing.
            progress.empty()
            st.error(f"הניתוח נכשל: {exc}")
            st.stop()

        progress.progress(100, text="הניתוח הסתיים")
        annotated_bytes = output_path.read_bytes()
        report_json = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)

        if result.status == "abstained":
            st.warning(result.message)
        else:
            st.success(result.message)

        st.subheader("הדגש של המאמן")
        if result.coaching.available:
            st.success(f"{result.coaching.focus}: {result.coaching.observation}")
        else:
            st.warning(result.coaching.observation)
        st.write(f"**Cue:** {result.coaching.cue}")
        st.write(f"**תרגיל:** {result.coaching.drill}")
        st.caption(f"ביטחון ההמלצה: {result.coaching.confidence:.0%}")

        cols = st.columns(4)
        cols[0].metric("ביטחון Pose", f"{result.pose_confidence:.0%}")
        cols[1].metric("זווית מרפק מינ׳", _fmt(result.metrics.min_elbow_angle_deg, "°"))
        cols[2].metric("זווית ברך מינ׳", _fmt(result.metrics.min_knee_angle_deg, "°"))
        cols[3].metric("עקביות גוף", _fmt(result.metrics.pose_consistency, "%", scale=100))

        detection_cols = st.columns(2)
        detection_cols[0].metric("זיהוי שחקן", f"{result.detections.player_detection_rate:.0%}")
        detection_cols[1].metric("זיהוי כדור", f"{result.detections.ball_detection_rate:.0%}")
        st.caption(
            "סל: "
            + (
                f"זוהה ({result.detections.rim_confidence:.0%})"
                if result.detections.rim_detected
                else "לא עבר את סף האמינות"
            )
        )

        release_cols = st.columns(2)
        release_cols[0].metric(
            "רגע שחרור",
            f"{result.release.seconds:.2f}s" if result.release else "לא זוהה",
        )
        release_cols[1].metric(
            "תוצאת הזריקה",
            {"made": "נכנסה", "missed": "החטאה"}.get(result.outcome.status, "לא ידוע"),
        )

        mechanics_cols = st.columns(3)
        mechanics_cols[0].metric(
            "גובה שחרור יחסי",
            _fmt(result.metrics.release_height_body_ratio, "× גוף"),
        )
        mechanics_cols[1].metric(
            "קפיצה יחסית",
            _fmt(result.metrics.jump_height_body_ratio, "%", scale=100),
        )
        mechanics_cols[2].metric(
            "סטיית נחיתה",
            _fmt(result.metrics.landing_drift_shoulder_widths, "× כתף"),
        )

        if result.quality.warnings:
            st.warning("איכות צילום: " + " · ".join(result.quality.warnings))

        st.subheader("כך המודל רואה את הסרטון")
        st.video(annotated_bytes)

        if result.event_window:
            st.caption(
                f"חלון התנועה המשוער: {result.event_window.start_seconds:.2f}–"
                f"{result.event_window.end_seconds:.2f} שניות. זהו איתור תנועה, לא עדיין "
                "זיהוי מאומת של שחרור הכדור."
            )

        with st.expander("דוח טכני מלא"):
            st.json(result.model_dump(mode="json"))

        download_video, download_report = st.columns(2)
        download_video.download_button(
            "הורד וידאו מסומן",
            data=annotated_bytes,
            file_name="swisher_annotated.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
        download_report.download_button(
            "הורד דוח JSON",
            data=report_json,
            file_name="swisher_report.json",
            mime="application/json",
            use_container_width=True,
        )
