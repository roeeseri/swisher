from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import streamlit as st

from swisher.annotation import DIAGNOSIS_LABELS_HE, AnnotationStore, next_shot_id
from swisher.data.io import atomic_write_json
from swisher.data.models import Outcome, Phase, ShotAnnotation, ShotEvents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_VIDEO_DIR = PROJECT_ROOT / "datasets" / "raw"
ANNOTATION_PATH = PROJECT_ROOT / "datasets" / "annotations" / "annotations.jsonl"
DRAFT_PATH = PROJECT_ROOT / "datasets" / "annotations" / ".current_draft.json"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}

PHASE_LABELS = {
    Phase.BASELINE.value: "לפני התערבות — Baseline",
    Phase.INTERVENTION.value: "בזמן התערבות המאמן",
    Phase.POST.value: "אחרי התערבות — Post",
}
OUTCOME_LABELS = {
    Outcome.MAKE.value: "נכנסה",
    Outcome.MISS.value: "החטאה",
    Outcome.UNKNOWN.value: "לא ניתן לקבוע",
}


def safe_filename(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(name).name)
    return clean[:120] or "uploaded_video.mp4"


def video_files() -> list[Path]:
    RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in RAW_VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def initialize_state() -> None:
    defaults: dict[str, object] = {
        "session_id": "",
        "video_id": "",
        "shot_id": "",
        "annotator_id": "",
        "phase": Phase.BASELINE.value,
        "outcome": Outcome.UNKNOWN.value,
        "diagnosis": "",
        "confidence": 3,
        "shot_start": 0,
        "load": 0,
        "release": 0,
        "rim": 0,
        "shot_end": 0,
        "evidence": "",
        "cue": "",
        "drill": "",
        "notes": "",
        "active_video": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_shot(records: list[ShotAnnotation]) -> None:
    video_id = str(st.session_state.video_id)
    for key in ("shot_start", "load", "release", "rim", "shot_end"):
        st.session_state[key] = 0
    for key in ("evidence", "cue", "drill", "notes", "diagnosis"):
        st.session_state[key] = ""
    st.session_state.outcome = Outcome.UNKNOWN.value
    st.session_state.confidence = 3
    st.session_state.shot_id = next_shot_id(records, video_id)


def load_record(record: ShotAnnotation) -> None:
    st.session_state.session_id = record.session_id
    st.session_state.video_id = record.video_id
    st.session_state.shot_id = record.shot_id
    st.session_state.annotator_id = record.annotator_id
    st.session_state.phase = record.phase.value
    st.session_state.outcome = record.outcome.value
    st.session_state.diagnosis = record.diagnosis_code or ""
    st.session_state.confidence = record.confidence
    st.session_state.shot_start = record.events.shot_start
    st.session_state.load = record.events.load or 0
    st.session_state.release = record.events.release or 0
    st.session_state.rim = record.events.rim_interaction or 0
    st.session_state.shot_end = record.events.shot_end
    st.session_state.evidence = record.evidence or ""
    st.session_state.cue = record.coach_cue or ""
    st.session_state.drill = record.drill_name or ""
    st.session_state.notes = record.notes or ""


def draft_payload() -> dict[str, object]:
    keys = [
        "session_id",
        "video_id",
        "shot_id",
        "annotator_id",
        "phase",
        "outcome",
        "diagnosis",
        "confidence",
        "shot_start",
        "load",
        "release",
        "rim",
        "shot_end",
        "evidence",
        "cue",
        "drill",
        "notes",
        "active_video",
    ]
    return {key: st.session_state.get(key) for key in keys}


st.set_page_config(
    page_title="Swisher Studio",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)
initialize_state()
store = AnnotationStore(ANNOTATION_PATH)
records = store.all()

st.markdown(
    """
    <style>
      .stApp { background: #f6f7f3; color: #17211b; }
      [data-testid="stSidebar"] { background: #10221a; }
      [data-testid="stSidebar"] * { color: #f5f7f2 !important; }
      .hero {
        background: linear-gradient(135deg,#10221a 0%,#1f5138 60%,#d9f45b 140%);
        border-radius: 22px; padding: 28px 32px; color: white; margin-bottom: 22px;
        box-shadow: 0 14px 40px rgba(16,34,26,.13);
      }
      .hero h1 { margin: 0; font-size: 2.1rem; letter-spacing: -.04em; }
      .hero p { margin: 8px 0 0; color: #dbe8df; font-size: 1.02rem; }
      .section-title { font-weight: 750; font-size: 1.1rem; margin: 8px 0 12px; }
      .status-card {
        background: white; border: 1px solid #e1e6df; border-radius: 16px;
        padding: 16px 18px; min-height: 92px;
      }
      .status-card small { color: #6b746e; }
      .status-card strong { display:block; font-size:1.35rem; margin-top:6px; }
      .stButton > button { border-radius: 12px; min-height: 44px; font-weight: 700; }
      div[data-baseweb="tab-list"] { gap: 8px; }
      button[data-baseweb="tab"] {
        background: white; border-radius: 12px; padding: 10px 18px;
      }
      .privacy-note {
        background:#eef3ed; border-right:4px solid #3d7251; border-radius:10px;
        padding:12px 14px; color:#34443a; font-size:.9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🏀 SWISHER")
    st.caption("COACH DATA STUDIO · PILOT")
    st.divider()
    st.text_input("מזהה מתייג", key="annotator_id", placeholder="coach_001")
    st.text_input("מזהה סשן", key="session_id", placeholder="session_001")
    st.text_input("מזהה וידאו", key="video_id", placeholder="video_001")
    st.selectbox(
        "שלב באימון",
        options=list(PHASE_LABELS),
        format_func=lambda value: PHASE_LABELS[value],
        key="phase",
    )
    st.divider()
    st.metric("זריקות שתויגו", len(records))
    known_outcomes = [record for record in records if record.outcome != Outcome.UNKNOWN]
    made = sum(record.outcome == Outcome.MAKE for record in known_outcomes)
    rate = made / len(known_outcomes) if known_outcomes else 0
    st.metric("אחוז קליעה מתויג", f"{rate:.0%}" if known_outcomes else "—")
    st.markdown(
        '<div class="privacy-note">השתמשו במזהים אנונימיים בלבד. '
        "אין להזין שמות מלאים או פרטים מזהים.</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="hero" dir="rtl">
      <h1>Swisher Coach Studio</h1>
      <p>הופכים וידאו מהאימון לדאטה מקצועי — זריקה אחת בכל פעם.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_a, metric_b, metric_c = st.columns(3)
with metric_a:
    st.markdown(
        f'<div class="status-card"><small>סשן פעיל</small><strong>'
        f'{st.session_state.session_id or "טרם הוגדר"}</strong></div>',
        unsafe_allow_html=True,
    )
with metric_b:
    st.markdown(
        f'<div class="status-card"><small>וידאו פעיל</small><strong>'
        f'{st.session_state.video_id or "טרם הוגדר"}</strong></div>',
        unsafe_allow_html=True,
    )
with metric_c:
    current = st.session_state.shot_id or next_shot_id(
        records, str(st.session_state.video_id)
    )
    st.markdown(
        f'<div class="status-card"><small>הזריקה הבאה</small><strong>'
        f"{current}</strong></div>",
        unsafe_allow_html=True,
    )

tab_video, tab_shot, tab_coach, tab_review = st.tabs(
    ["1 · בחירת וידאו", "2 · סימון הזריקה", "3 · אבחנת המאמן", "4 · בדיקה ושמירה"]
)

with tab_video:
    st.markdown('<div class="section-title">בחירת חומר גלם</div>', unsafe_allow_html=True)
    source_existing, source_upload = st.columns([1.15, 0.85])
    with source_existing:
        available = video_files()
        labels = ["— בחרו סרטון —"] + [path.name for path in available]
        selected_name = st.selectbox("סרטונים בתיקיית raw", labels)
        if selected_name != labels[0]:
            st.session_state.active_video = str(RAW_VIDEO_DIR / selected_name)
        if st.session_state.active_video and Path(st.session_state.active_video).is_file():
            st.video(st.session_state.active_video)
            st.caption(Path(st.session_state.active_video).name)
        elif not available:
            st.info("עדיין אין סרטונים. העלו סרטון ראשון בצד שמאל.")
    with source_upload:
        uploaded = st.file_uploader(
            "העלאת סרטון חדש",
            type=["mp4", "mov", "m4v"],
            help="הקובץ מועתק ל־datasets/raw; המקור אינו משתנה.",
        )
        if uploaded is not None:
            target = RAW_VIDEO_DIR / safe_filename(uploaded.name)
            st.caption(f"הקובץ יישמר בשם: {target.name}")
            if target.exists():
                st.warning("קיים קובץ בשם זה. שנו את שם הקובץ לפני ההעלאה.")
            elif st.button("ייבוא הסרטון", type="primary", use_container_width=True):
                target.write_bytes(uploaded.getbuffer())
                st.session_state.active_video = str(target)
                st.success("הסרטון יובא בהצלחה.")
                st.rerun()
    st.info(
        "לפיילוט: מצלמה אחורית, חצובה, 1080p/60fps, וכל הגוף + הכדור + הסל בפריים."
    )

with tab_shot:
    st.markdown(
        '<div class="section-title">אירועי הזריקה בפריימים</div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.active_video:
        st.warning("בחרו קודם סרטון בלשונית הראשונה.")
    st.text_input("מזהה זריקה", key="shot_id", placeholder=current)
    st.caption("אם אירוע אינו נראה בוודאות, השאירו 0. המערכת לא תנחש במקומכם.")
    frame_columns = st.columns(5)
    frame_columns[0].number_input("תחילת זריקה", min_value=0, step=1, key="shot_start")
    frame_columns[1].number_input("Load", min_value=0, step=1, key="load")
    frame_columns[2].number_input("Release", min_value=0, step=1, key="release")
    frame_columns[3].number_input("מגע בטבעת", min_value=0, step=1, key="rim")
    frame_columns[4].number_input("סיום זריקה", min_value=0, step=1, key="shot_end")
    st.radio(
        "תוצאת הזריקה",
        options=list(OUTCOME_LABELS),
        format_func=lambda value: OUTCOME_LABELS[value],
        horizontal=True,
        key="outcome",
    )

with tab_coach:
    st.markdown('<div class="section-title">מה המאמן ראה?</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        diagnoses = [""] + list(DIAGNOSIS_LABELS_HE)
        st.selectbox(
            "האבחנה המרכזית — אחת בלבד",
            diagnoses,
            format_func=lambda value: (
                "ללא אבחנה" if not value else DIAGNOSIS_LABELS_HE[value]
            ),
            key="diagnosis",
        )
        st.text_area(
            "מה ראית שגרם לאבחנה?",
            key="evidence",
            placeholder="לדוגמה: בנחיתה השחקן זז שמאלה ב־4 מתוך 5 זריקות",
        )
        st.slider(
            "רמת ביטחון באבחנה",
            min_value=1,
            max_value=5,
            help="1 = השערה חלשה, 5 = ברור מאוד",
            key="confidence",
        )
    with right:
        st.text_input(
            "הדגש שנאמר לשחקן",
            key="cue",
            placeholder='לדוגמה: "סיים מאוזן ונחת באותו מקום"',
        )
        st.text_input("שם התרגיל", key="drill", placeholder="לדוגמה: Form shooting")
        st.text_area("הערות נוספות", key="notes")

with tab_review:
    st.markdown(
        '<div class="section-title">בדיקה לפני שמירה</div>',
        unsafe_allow_html=True,
    )
    required = {
        "מזהה מתייג": st.session_state.annotator_id,
        "מזהה סשן": st.session_state.session_id,
        "מזהה וידאו": st.session_state.video_id,
        "מזהה זריקה": st.session_state.shot_id,
        "סרטון": st.session_state.active_video,
    }
    missing = [label for label, value in required.items() if not value]
    summary_left, summary_right = st.columns(2)
    with summary_left:
        st.write("**פרטי הזריקה**")
        st.write(f"סשן: `{st.session_state.session_id or '—'}`")
        st.write(f"וידאו: `{st.session_state.video_id or '—'}`")
        st.write(f"זריקה: `{st.session_state.shot_id or '—'}`")
        st.write(f"תוצאה: **{OUTCOME_LABELS[st.session_state.outcome]}**")
    with summary_right:
        st.write("**אבחנת המאמן**")
        diagnosis_value = st.session_state.diagnosis
        st.write(
            "אבחנה: **"
            + DIAGNOSIS_LABELS_HE.get(diagnosis_value, "ללא אבחנה")
            + "**"
        )
        st.write(f"דגש: {st.session_state.cue or '—'}")
        st.write(f"ביטחון: {st.session_state.confidence}/5")
    if missing:
        st.error("חסרים פרטים: " + ", ".join(missing))
    event_values = [
        int(st.session_state.shot_start),
        int(st.session_state.load) or None,
        int(st.session_state.release) or None,
        int(st.session_state.rim) or None,
        int(st.session_state.shot_end),
    ]
    known_events = [value for value in event_values if value is not None]
    if known_events != sorted(known_events):
        st.error("סדר הפריימים אינו כרונולוגי. חזרו ללשונית סימון הזריקה.")
    save_col, next_col = st.columns(2)
    with save_col:
        save_clicked = st.button(
            "שמירת הזריקה",
            type="primary",
            use_container_width=True,
            disabled=bool(missing),
        )
    with next_col:
        if st.button("ניקוי ומעבר לזריקה הבאה", use_container_width=True):
            reset_shot(records)
            st.rerun()
    if save_clicked:
        try:
            record = ShotAnnotation(
                annotation_id=str(uuid4()),
                session_id=str(st.session_state.session_id),
                video_id=str(st.session_state.video_id),
                shot_id=str(st.session_state.shot_id),
                annotator_id=str(st.session_state.annotator_id),
                events=ShotEvents(
                    shot_start=int(st.session_state.shot_start),
                    load=int(st.session_state.load) or None,
                    release=int(st.session_state.release) or None,
                    rim_interaction=int(st.session_state.rim) or None,
                    shot_end=int(st.session_state.shot_end),
                ),
                outcome=Outcome(st.session_state.outcome),
                diagnosis_code=str(st.session_state.diagnosis) or None,
                evidence=str(st.session_state.evidence) or None,
                coach_cue=str(st.session_state.cue) or None,
                drill_name=str(st.session_state.drill) or None,
                phase=Phase(st.session_state.phase),
                confidence=int(st.session_state.confidence),
                notes=str(st.session_state.notes) or None,
            )
            store.upsert(record)
            st.success("הזריקה נשמרה. JSONL ו־CSV עודכנו.")
            st.balloons()
        except Exception as error:
            st.error(f"הזריקה לא נשמרה: {error}")

st.divider()
st.subheader("היסטוריית תיוגים")
if not records:
    st.caption("עדיין לא נשמרו זריקות.")
else:
    filtered = [
        record
        for record in records
        if not st.session_state.session_id
        or record.session_id == st.session_state.session_id
    ]
    for record in reversed(filtered[-20:]):
        label = (
            f"{record.shot_id} · {OUTCOME_LABELS[record.outcome.value]} · "
            f"{DIAGNOSIS_LABELS_HE.get(record.diagnosis_code or '', 'ללא אבחנה')}"
        )
        with st.expander(label):
            st.caption(
                f"{record.video_id} · גרסת תיוג {record.annotation_version} · "
                f"ביטחון {record.confidence}/5"
            )
            st.write(record.coach_cue or "לא הוזן דגש")
            if st.button("פתיחת התיוג לעריכה", key=f"edit_{record.annotation_id}"):
                load_record(record)
                st.rerun()

try:
    atomic_write_json(draft_payload(), DRAFT_PATH)
except OSError:
    st.caption("לא ניתן לשמור טיוטה אוטומטית כרגע.")
