from __future__ import annotations

import streamlit as st


def theme() -> None:
    st.markdown(
        """<style>
    .stApp{direction:rtl;background:#0b1220;color:#eef3fb}
    [data-testid="stSidebar"]{background:#101d30}
    .block-container{max-width:1320px;padding-top:2rem}
    h1,h2,h3{letter-spacing:-.025em}
    .hero{padding:30px;border:1px solid #30435e;border-radius:20px;
      background:linear-gradient(120deg,#1b3150,#142136);margin-bottom:22px}
    .hero .eyebrow{color:#f9a46b;font-size:13px;letter-spacing:.16em;direction:ltr;text-align:right}
    .hero h1{font-size:42px;margin:10px 0;color:#fff}
    .hero p{color:#c7d5e8;font-size:18px;margin:0;line-height:1.7}
    [data-testid="stMetric"]{background:#132137;border:1px solid #283c58;
      padding:18px;border-radius:13px}
    [data-testid="stMetricLabel"]{color:#b8c9e0}
    .stButton>button,.stDownloadButton>button{border-radius:10px;min-height:44px}
    [data-testid="stTabs"] button{font-size:17px}
    [data-testid="stFileUploader"]{direction:rtl}
    [data-testid="stNumberInput"] input{direction:ltr}
    .stCaption{color:#b8c9e0}code{direction:ltr}
    </style>""",
        unsafe_allow_html=True,
    )


def report_view(report: dict) -> None:
    cols = st.columns(3)
    cols[0].metric("כיסוי גוף", f"{report['pose_confidence']:.0%}")
    cols[1].metric("כיסוי כדור בדגימות", f"{report['detections']['ball_detection_rate']:.0%}")
    release = report.get("release")
    cols[2].metric("שחרור משוער", f"{release['seconds']:.2f} שנ׳" if release else "דורש תיוג")
    st.caption("כיסוי = שיעור הפריימים שבהם התקבל זיהוי. זה אינו אחוז דיוק.")
    if report["status"] != "ok":
        st.warning(report["message"])
    for warning in report["quality"]["warnings"]:
        st.warning(warning)
    coaching = report["coaching"]
    st.subheader("דגש מוצע לבדיקה עם המאמן")
    st.write(coaching["observation"])
    if coaching["available"]:
        st.info(coaching["cue"])
        st.write(coaching["drill"])
        for evidence in coaching["evidence"]:
            st.caption(evidence)
    st.caption("הצעה ניסיונית; אין הוכחה שהיא תשפר קליעה. אישור המאמן נשמר בנפרד.")
    with st.expander("ראיות ומדידות מחקריות · לא אבחון"):
        if report["status"] == "ok":
            values = {
                k: round(v, 2)
                for k, v in report["metrics"].items()
                if isinstance(v, (int, float)) and k != "pose_consistency"
            }
            st.json(values)
            st.caption("זוויות הן היטל דו־ממדי, ללא מדידת כוח, סיבוב כדור או גובה קפיצה.")
        else:
            st.write("מדדים מוסתרים כאשר כיסוי הגוף אינו מספיק.")
        st.json(report.get("provenance", {}))
