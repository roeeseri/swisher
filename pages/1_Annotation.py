from __future__ import annotations

from pathlib import Path

import streamlit as st

from swisher.session import SessionStore, ShotAnnotation, read_frame

ROOT = Path(__file__).resolve().parents[1]
STORE = SessionStore(ROOT / "data" / "sessions")

PHASE_LABELS = {
    "baseline": "לפני התערבות — Baseline",
    "intervention": "התערבות המאמן",
    "post_intervention": "אחרי ההתערבות — Post",
}
OUTCOME_LABELS = {"made": "נכנסה", "missed": "החטאה", "unknown": "לא ידוע"}
DIAGNOSES = {
    "none": "ללא אבחנה",
    "balance": "חוסר איזון",
    "leg_usage": "שימוש לא עקבי ברגליים",
    "elbow_alignment": "יישור מרפק לא עקבי",
    "release_timing": "תזמון שחרור לא עקבי",
    "follow_through": "Follow-through קצר",
    "landing_drift": "סטייה בנחיתה",
}

st.set_page_config(page_title="Swisher Session Studio", page_icon="🏷️", layout="wide")
st.markdown(
    """
    <style>
      .stApp { direction: rtl; background: #f7f8f5; }
      [data-testid="stSidebar"] { background:#10221a; }
      [data-testid="stSidebar"] * { color:white !important; }
      .hero {background:linear-gradient(135deg,#10221a,#27613f);color:white;
             padding:24px 28px;border-radius:20px;margin-bottom:20px}
      .hero h1 {margin:0}.hero p {margin:8px 0 0;color:#dce9df}
    </style>
    <div class="hero"><h1>Swisher Session Studio</h1>
    <p>שמירת סשן רציף, סימון זריקות ובניית אמת הקרקע של Swisher.</p></div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🏀 SWISHER")
    st.caption("GOLDEN DATASET · PILOT")
    st.info("אין להזין שמות מלאים. השתמשו במזהים אנונימיים בלבד.")

create_tab, annotate_tab, history_tab = st.tabs(["1 · סשן חדש", "2 · סימון זריקות", "3 · מצב הסשן"])

with create_tab:
    st.subheader("פתיחת סשן צילום")
    uploaded = st.file_uploader(
        "העלה סרטון רציף של הסשן",
        type=["mp4", "mov", "m4v", "avi"],
        help="מומלץ: 1080p, 60FPS, חצובה וכל הגוף והסל בפריים.",
    )
    first, second = st.columns(2)
    with first:
        new_session_id = st.text_input("מזהה סשן", placeholder="session_2026_001")
        new_player_id = st.text_input("מזהה שחקן", placeholder="player_001")
        new_coach_id = st.text_input("מזהה מאמן", placeholder="coach_001")
    with second:
        phase = st.selectbox("שלב האימון", list(PHASE_LABELS), format_func=PHASE_LABELS.get)
        camera_view = st.selectbox(
            "זווית צילום",
            ["side_oblique", "front_oblique", "side", "front", "other"],
        )
        consent = st.checkbox("קיימת הסכמה מתועדת לצילום ולשימוש בדאטה")
    can_create = all([uploaded, new_session_id, new_player_id, new_coach_id, consent])
    if st.button(
        "צור סשן ושמור את הסרטון",
        type="primary",
        use_container_width=True,
        disabled=not can_create,
    ):
        try:
            metadata = STORE.create(
                session_id=new_session_id,
                player_id=new_player_id,
                coach_id=new_coach_id,
                phase=phase,
                camera_view=camera_view,
                consent_confirmed=consent,
                video_name=uploaded.name,
                video_bytes=uploaded.getvalue(),
            )
            st.session_state["active_session"] = metadata.session_id
            st.success(
                f"הסשן נשמר: {metadata.frame_count} פריימים, "
                f"{metadata.fps:.1f} FPS, {metadata.width}×{metadata.height}."
            )
        except Exception as exc:
            st.error(f"הסשן לא נשמר: {exc}")

with annotate_tab:
    sessions = STORE.list_sessions()
    if not sessions:
        st.info("צור קודם סשן בלשונית הראשונה.")
    else:
        default_session = st.session_state.get("active_session", sessions[-1])
        default_index = sessions.index(default_session) if default_session in sessions else 0
        selected_session = st.selectbox("בחר סשן", sessions, index=default_index)
        metadata = STORE.load(selected_session)
        video_path = STORE.video_path(selected_session)
        annotations = STORE.annotations(selected_session)
        st.video(str(video_path))
        st.caption(
            f"{metadata.player_id} · {PHASE_LABELS[metadata.phase]} · "
            f"{metadata.frame_count} פריימים · {metadata.fps:.1f} FPS"
        )

        st.subheader("מצא את רגעי הזריקה")
        preview_frame = st.slider(
            "פריים לתצוגה",
            0,
            metadata.frame_count - 1,
            0,
            help="גרור כדי לבדוק את הווידאו פריים אחר פריים.",
        )
        try:
            st.image(
                read_frame(video_path, preview_frame),
                caption=f"פריים {preview_frame} · {preview_frame / metadata.fps:.2f}s",
                use_container_width=True,
            )
        except ValueError as exc:
            st.warning(str(exc))

        next_number = len(annotations) + 1
        with st.form("shot_annotation", clear_on_submit=True):
            shot_id = st.text_input("מזהה זריקה", value=f"shot_{next_number:03d}")
            event_cols = st.columns(4)
            shot_start = event_cols[0].number_input(
                "תחילת זריקה", 0, metadata.frame_count - 1, preview_frame
            )
            release_frame = event_cols[1].number_input(
                "שחרור", 0, metadata.frame_count - 1, preview_frame
            )
            rim_value = event_cols[2].number_input(
                "מגע בסל (0=לא ידוע)", 0, metadata.frame_count - 1, 0
            )
            shot_end = event_cols[3].number_input(
                "סיום זריקה", 0, metadata.frame_count - 1, preview_frame
            )
            coach_left, coach_right = st.columns(2)
            with coach_left:
                outcome = st.radio(
                    "תוצאה",
                    list(OUTCOME_LABELS),
                    format_func=OUTCOME_LABELS.get,
                    horizontal=True,
                )
                diagnosis = st.selectbox("אבחנה מרכזית", list(DIAGNOSES), format_func=DIAGNOSES.get)
                confidence = st.slider("ביטחון המאמן", 1, 5, 3)
            with coach_right:
                evidence = st.text_area("מה המאמן ראה?", placeholder="תאר סימן שאפשר לראות בווידאו")
                cue = st.text_input("הדגש שניתן לשחקן")
                drill = st.text_input("התרגיל שניתן")
            submitted = st.form_submit_button(
                "שמור זריקה", type="primary", use_container_width=True
            )
        if submitted:
            try:
                annotation = ShotAnnotation(
                    shot_id=shot_id,
                    shot_start=int(shot_start),
                    release_frame=int(release_frame),
                    rim_interaction=int(rim_value) or None,
                    shot_end=int(shot_end),
                    outcome=outcome,
                    diagnosis=diagnosis,
                    coach_evidence=evidence,
                    coach_cue=cue,
                    drill=drill,
                    confidence=confidence,
                )
                STORE.upsert_annotation(selected_session, annotation)
                st.success(f"{shot_id} נשמרה. אפשר לעבור לזריקה הבאה.")
                st.rerun()
            except Exception as exc:
                st.error(f"הזריקה לא נשמרה: {exc}")

        if annotations:
            st.subheader("זריקות שכבר סומנו")
            st.dataframe(
                [
                    {
                        "זריקה": item.shot_id,
                        "התחלה": item.shot_start,
                        "שחרור": item.release_frame,
                        "סיום": item.shot_end,
                        "תוצאה": OUTCOME_LABELS[item.outcome],
                        "אבחנה": DIAGNOSES.get(item.diagnosis, item.diagnosis),
                        "גרסה": item.annotation_version,
                    }
                    for item in annotations
                ],
                use_container_width=True,
                hide_index=True,
            )

with history_tab:
    sessions = STORE.list_sessions()
    if not sessions:
        st.caption("עדיין אין סשנים.")
    for session_id in reversed(sessions):
        metadata = STORE.load(session_id)
        annotations = STORE.annotations(session_id)
        analyzed = sum(
            (STORE.artifact_directory(session_id, item.shot_id) / "report.json").exists()
            for item in annotations
        )
        with st.expander(f"{session_id} · {len(annotations)} זריקות"):
            st.write(
                f"שחקן: `{metadata.player_id}` · מאמן: `{metadata.coach_id}` · "
                f"נותחו: {analyzed}/{len(annotations)}"
            )
            st.progress(analyzed / len(annotations) if annotations else 0)
