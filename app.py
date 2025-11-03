import streamlit as st
import pandas as pd
from datetime import datetime
import os
import plotly.express as px
from openai import OpenAI
import random

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Employee Sentiment Tracker", page_icon="💬", layout="wide")
st.title("💬 Employee Sentiment Tracker")
st.caption("POC – AI-powered employee feedback analysis & adaptive survey generation")

# ---------- OPENAI CLIENT ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# ---------- SENTIMENT NORMALIZATION ----------
ALLOWED_SENTIMENTS = {"Positive", "Negative", "Neutral", "Frustrated"}

def normalize_sentiment(val: str) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    t = str(val).strip().lower()
    if "frustrat" in t or t in {"angry", "annoyed"}:
        return "Frustrated"
    if "neg" in t or "bad" in t or "issue" in t:
        return "Negative"
    if "posit" in t or t in {"good", "great", "happy", "awesome"}:
        return "Positive"
    if "neutral" in t:
        return "Neutral"
    cap = t.title()
    return cap if cap in ALLOWED_SENTIMENTS else ""

def clean_df_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].apply(normalize_sentiment)
        df = df[df["sentiment"].isin(ALLOWED_SENTIMENTS) | (df["sentiment"] == "")]
    return df

# ---------- HELPERS ----------
@st.cache_data
def load_data():
    if os.path.exists("feedback.csv"):
        df = pd.read_csv("feedback.csv")
        return clean_df_sentiment(df)
    else:
        return pd.DataFrame(columns=["id", "date", "employee", "department", "message", "sentiment", "topic"])

def save_data(df):
    df.drop_duplicates(subset="id", keep="last", inplace=True)
    df = clean_df_sentiment(df)
    df.to_csv("feedback.csv", index=False)

# ---------- SESSION STATE ----------
if "df" not in st.session_state:
    st.session_state.df = load_data()
    save_data(st.session_state.df)

# ---------- LOAD LATEST QUESTIONS ----------
default_questions = [
    "How do you feel about your workload recently?",
    "How supported do you feel by your manager or team?",
    "Any suggestions to improve your work environment?"
]

latest_qs = default_questions
if os.path.exists("questions_history.csv"):
    try:
        hist_df = pd.read_csv("questions_history.csv")
        if not hist_df.empty:
            last_q_str = hist_df.iloc[-1]["questions"]
            latest_qs = [q.strip() for q in last_q_str.split("|") if q.strip()]
    except Exception as e:
        st.warning(f"⚠️ Could not read questions history: {e}")

if "questions" not in st.session_state:
    st.session_state.questions = latest_qs

df = st.session_state.df

# ---------- LOCAL CLASSIFIERS ----------
def local_sentiment(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["angry", "frustrated", "upset", "annoyed", "sick of", "tired of"]):
        return "Frustrated"
    if any(w in t for w in ["cannot", "not working", "slow", "error", "bad", "fail", "problem", "issue"]):
        return "Negative"
    if any(w in t for w in ["thank", "thanks", "good", "great", "appreciate", "helpful", "love", "happy", "awesome"]):
        return "Positive"
    return "Neutral"

def local_topic(text: str) -> str:
    t = text.lower()
    if "workload" in t or "busy" in t or "task" in t:
        return "Workload"
    if "manager" in t or "support" in t or "leader" in t or "boss" in t:
        return "Management Support"
    if "office" in t or "environment" in t or "remote" in t:
        return "Work Environment"
    if "communication" in t or "meeting" in t:
        return "Communication"
    if "career" in t or "training" in t or "growth" in t:
        return "Growth"
    return "Others"

# ---------- ADAPTIVE QUESTION GENERATOR ----------
st.subheader("🧩 Generate Next Questionnaire")

if client and st.button("Generate next questionnaire"):
    sentiment_summary = df["sentiment"].value_counts().to_dict()
    top_topics = df["topic"].value_counts().nlargest(3).index.tolist()
    sample_texts = "\n".join(df["message"].tail(10).tolist()) if not df.empty else "No feedback yet."

    prompt = f"""
    You are an HR assistant creating adaptive employee engagement questionnaires.
    Recent feedback:
    {sample_texts}
    Sentiment mix: {sentiment_summary}.
    Top themes: {', '.join(top_topics)}.

    Generate 5 short open-ended questions (under 20 words).
    Focus on areas where morale is low (Negative or Frustrated sentiment).
    Include one positive reflection question.
    Number them 1–5.
    """

    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        q_text = resp.choices[0].message.content
        new_qs = [line[line.find(".")+1:].strip() for line in q_text.splitlines() if line.strip() and line.strip()[0].isdigit()]
        if new_qs:
            st.session_state.questions = new_qs
            latest_qs = " | ".join(st.session_state.questions)
            hist_df = pd.read_csv("questions_history.csv") if os.path.exists("questions_history.csv") else pd.DataFrame(columns=["timestamp", "questions"])
            new_entry = pd.DataFrame({"timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], "questions": [latest_qs]})
            hist_df = pd.concat([hist_df, new_entry], ignore_index=True)
            hist_df.to_csv("questions_history.csv", index=False)
            st.success("✅ New questionnaire saved!")
            st.rerun()
    except Exception as e:
        st.error(f"⚠️ OpenAI error: {e}")

# ---------- LAYOUT ----------
st.markdown("---")
left_col, right_col = st.columns([1.2, 1])

# LEFT: FEEDBACK FORM
with left_col:
    st.subheader("📝 Employee Feedback Form")
    with st.form("employee_form", clear_on_submit=True):
        name = st.text_input("Employee Name (optional)")
        dept = st.selectbox(
            "Division / Department",
            ["", "FSC", "SSO", "Crisis Shelter", "Transitional Shelter", "Care Staff", "Welfare Officer", "Others"]
        )
        answers = [st.text_area(q, height=80) for q in st.session_state.questions]
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            msg = " | ".join([a for a in answers if a.strip()])
            if msg:
                with st.spinner("🔍 Analyzing feedback..."):
                    if client:
                        try:
                            prompt = f"Classify employee feedback. Output CSV only: sentiment,topic. Message: {msg.strip()}"
                            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
                            output = resp.choices[0].message.content.strip()
                            parts = [p.strip() for p in output.split(",")]
                            sentiment = normalize_sentiment(parts[0]) if len(parts) > 0 else local_sentiment(msg)
                            topic = parts[1] if len(parts) > 1 else local_topic(msg)
                        except Exception:
                            sentiment = local_sentiment(msg)
                            topic = local_topic(msg)
                    else:
                        sentiment = local_sentiment(msg)
                        topic = local_topic(msg)

                new_row = {
                    "id": int(df["id"].max()) + 1 if len(df) else 1,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "employee": name,
                    "department": dept,
                    "message": msg.strip(),
                    "sentiment": sentiment,
                    "topic": topic
                }
                st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
                save_data(st.session_state.df)
                st.success(f"✅ Saved — Sentiment: **{sentiment}**, Topic: **{topic}**")
                st.rerun()

# RIGHT: HISTORY + UPLOAD + DEMO CSV
with right_col:
    st.subheader("📜 Questionnaire & Data Management")

    # Questionnaire history
    if os.path.exists("questions_history.csv"):
        hist = pd.read_csv("questions_history.csv")
        if not hist.empty:
            st.markdown("**🕓 Questionnaire History:**")
            hist_display = hist.copy()
            hist_display.index = range(1, len(hist_display) + 1)
            hist_display.rename(columns={"timestamp": "Generated At", "questions": "Questions"}, inplace=True)
            st.dataframe(hist_display.iloc[::-1], use_container_width=True)
        else:
            st.caption("No questionnaires generated yet.")

    st.download_button("⬇️ Download Current Questions (CSV)",
                       data="\n".join(st.session_state.questions),
                       file_name="current_questions.csv",
                       mime="text/csv")

    st.markdown("### 📤 Upload Bulk Feedback CSV")
    uploaded = st.file_uploader("Upload CSV (id,date,employee,department,message)", type=["csv"])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            required_cols = {"id", "date", "employee", "department", "message"}
            if not required_cols.issubset(new_df.columns):
                st.error(f"⚠️ Missing required columns: {', '.join(required_cols)}")
            else:
                new_df["sentiment"], new_df["topic"] = "", ""
                if client:
                    st.info(f"🧠 Classifying {len(new_df)} rows with AI...")
                    for idx, row in new_df.iterrows():
                        msg = row["message"]
                        try:
                            prompt = f"Classify employee feedback. Output CSV only: sentiment,topic. Message: {msg.strip()}"
                            resp = client.chat.completions.create(model="gpt-4o-mini",
                                                                  messages=[{"role": "user", "content": prompt}])
                            output = resp.choices[0].message.content.strip()
                            parts = [p.strip() for p in output.split(",")]
                            new_df.at[idx, "sentiment"] = normalize_sentiment(parts[0]) if len(parts) > 0 else local_sentiment(msg)
                            new_df.at[idx, "topic"] = parts[1] if len(parts) > 1 else local_topic(msg)
                        except Exception:
                            new_df.at[idx, "sentiment"] = local_sentiment(msg)
                            new_df.at[idx, "topic"] = local_topic(msg)
                    st.success("✅ AI classification complete.")
                else:
                    new_df["sentiment"] = new_df["message"].apply(local_sentiment)
                    new_df["topic"] = new_df["message"].apply(local_topic)

                combined = pd.concat([st.session_state.df, new_df], ignore_index=True)
                save_data(combined)
                st.session_state.df = combined
                st.success(f"✅ Uploaded and merged {len(new_df)} new responses.")
                st.rerun()
        except Exception as e:
            st.error(f"⚠️ Error reading CSV: {e}")

    # Demo CSV Generator
    st.markdown("### 🧪 Demo CSV Generator")
    if st.button("Generate Demo CSV"):
        demo_employees = ["John Tan", "Maria Lim", "Ahmad Yusof", "Priya Menon", "Wei Ming"]
        demo_depts = ["FSC", "SSO", "Crisis Shelter", "Transitional Shelter", "Care Staff", "Welfare Officer"]
        demo_responses = []

        for i in range(10):
            emp = random.choice(demo_employees)
            dept = random.choice(demo_depts)
            q = random.choice(st.session_state.questions)
            msg = f"{emp} feedback ({dept}): {q.lower()} - " + random.choice([
                "I feel my workload has increased a lot recently.",
                "My manager has been very supportive and approachable.",
                "The new workflow system often crashes during peak hours.",
                "The communication between shifts could be clearer.",
                "I enjoy the teamwork but sometimes feel understaffed.",
                "More opportunities for career growth would be appreciated."
            ])
            demo_responses.append({
                "id": i + 1,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "employee": emp,
                "department": dept,
                "message": msg
            })

        demo_df = pd.DataFrame(demo_responses)
        demo_df.to_csv("demo_feedback.csv", index=False)

        st.success("✅ Demo CSV file generated successfully.")
        csv_data = demo_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Demo Feedback CSV",
            data=csv_data,
            file_name="demo_feedback.csv",
            mime="text/csv"
        )
