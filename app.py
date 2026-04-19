import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.agent import MentalHealthAgent

st.set_page_config(page_title="Mental Health Screening", page_icon="🧠", layout="centered")

st.title("🧠 Mental Health Screening System")
st.markdown("This autonomous agent screens for **stress** and **depression** based on your responses.")
st.markdown("---")

@st.cache_resource
def load_agent():
    return MentalHealthAgent()

agent = load_agent()

tab1, tab2 = st.tabs(["Stress Screening", "Depression Screening"])

# ── STRESS TAB ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Stress Screening Questionnaire")
    st.caption("Rate each factor from 1 to 5")

    sleep_quality       = st.slider("Sleep quality (1=very poor, 5=excellent)",        1, 5, 3)
    headaches_weekly    = st.slider("Headaches per week (1=none, 5=daily)",             1, 5, 2)
    academic_performance= st.slider("Academic performance (1=struggling, 5=excellent)", 1, 5, 3)
    study_load          = st.slider("Study load (1=light, 5=overwhelming)",             1, 5, 3)
    extracurricular     = st.slider("Extracurricular activities (1=few, 5=many)",       1, 5, 2)

    if st.button("Run Stress Screening", type="primary"):
        features = {
            'sleep_quality':          sleep_quality,
            'headaches_weekly':       headaches_weekly,
            'academic_performance':   academic_performance,
            'study_load':             study_load,
            'extracurricular_weekly': extracurricular,
        }

        risk, label, confidence = agent.assess_stress_risk(features)
        action = agent.decide_action(risk)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            color_map = {"Low Stress": "green", "Moderate Stress": "orange", "High Stress": "red"}
            color = color_map.get(label, "orange")
            st.markdown(f"### :{color}[{label}]")
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")

        st.markdown("**Agent reasoning:**")
        risk_score = (features['sleep_quality'] * 0.3 + features['headaches_weekly'] * 0.25 + features['study_load'] * 0.25 + (5 - features['academic_performance']) * 0.1 + (5 - features['extracurricular_weekly']) * 0.1) 
        st.progress(risk_score / 5.0, text=f"Risk score: {risk_score:.2f} / 5.0")

        st.markdown("**Agent decision:** `" + action.replace("_", " ") + "`")
        st.markdown("---")

        if action == "provide_reassurance":
            st.success("Your stress levels appear manageable. Keep up good sleep hygiene and time management.")
        elif action == "recommend_resources":
            st.warning("You're experiencing moderate stress. Here are some resources:")
            st.markdown("""
- Deep breathing exercises (5 min, 3x daily)
- Set realistic daily goals
- Take short walks between study sessions
- **Mindfulness Apps:** Headspace or Calm (free for students)
- **Crisis Text Line:** Text HOME to 741741
            """)
        else:
            st.error("Your responses indicate high stress. Please reach out for support.")
            st.markdown("""
- **University Counseling Center:** Schedule a free confidential appointment
- **National Helpline:** 988 (Suicide & Crisis Lifeline)
- **Crisis Text Line:** Text HOME to 741741
            """)

        agent.tools['log_interaction'](
            {'risk_level': risk, 'label': label, 'action': action},
            action, f"Web UI screening | confidence={confidence:.2f}"
        )
        st.caption("Session logged.")

# ── DEPRESSION TAB ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("PHQ-9 Depression Screening")
    st.caption("Over the last 2 weeks, how often have you been bothered by the following?")

    options = ["Not at all (0)", "Several days (1)", "More than half the days (2)", "Nearly every day (3)"]
    score_map = {o: i for i, o in enumerate(options)}

    q1  = st.selectbox("Little interest or pleasure in doing things", options)
    q2  = st.selectbox("Feeling down, depressed, or hopeless", options)
    q3  = st.selectbox("Trouble falling or staying asleep, or sleeping too much", options)
    q4  = st.selectbox("Feeling tired or having little energy", options)
    q5  = st.selectbox("Poor appetite or overeating", options)
    q6  = st.selectbox("Feeling bad about yourself or that you are a failure", options)
    q7  = st.selectbox("Trouble concentrating on things", options)
    q8  = st.selectbox("Moving or speaking so slowly others could notice / being fidgety or restless", options)
    q9  = st.selectbox("Thoughts that you would be better off dead or hurting yourself", options)

    if st.button("Run Depression Screening", type="primary"):
        responses = [score_map[q] for q in [q1,q2,q3,q4,q5,q6,q7,q8,q9]]
        total = sum(responses)

        if   total <= 4:  severity, color = "Minimal",            "green"
        elif total <= 9:  severity, color = "Mild",               "green"
        elif total <= 14: severity, color = "Moderate",           "orange"
        elif total <= 19: severity, color = "Moderately Severe",  "orange"
        else:             severity, color = "Severe",             "red"

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### :{color}[{severity} Depression]")
        with col2:
            st.metric("PHQ-9 Score", f"{total} / 27")

        st.progress(total / 27, text=f"Total score: {total}")

        if total <= 4:
            st.success("Minimal symptoms detected. Maintain healthy habits.")
        elif total <= 9:
            st.info("Mild symptoms. Consider self-care strategies and monitoring.")
        elif total <= 14:
            st.warning("Moderate symptoms. Talking to a counselor is recommended.")
        else:
            st.error("Significant symptoms detected. Please seek professional support.")
            st.markdown("""
- **University Counseling Center:** Free confidential appointments
- **National Helpline:** 988
- **Crisis Text Line:** Text HOME to 741741
            """)

        if score_map[q9] >= 2:
            st.error("You indicated thoughts of self-harm. Please reach out immediately — you are not alone.")

        st.caption(f"PHQ-9 score: {total}/27 | Severity: {severity}")
