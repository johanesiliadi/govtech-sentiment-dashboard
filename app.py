import streamlit as st
import pandas as pd
from datetime import datetime
import os
import plotly.express as px
from openai import OpenAI

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
    t = str(val).strip().lower().replace(";", "").replace(",", "").replace(".", "")
    if t in {"frustration", "frustrated", "angry", "annoyed"} or "frustrat" in t:
        return "Frustrated"
    if t in {"neg", "negative", "bad"}:
        return "Negative"
    if t.startswith("posit") or t in {"good", "great", "happy", "awesome", "fantastic", "excellent"}:
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
    if any(w in t for w in ["angry", "frustrated", "upset", "annoyed", "sick of", "fed up", "ridiculous", "every time", "always", "again and again", "tired of"]):
        return "Frustrated"
    if any(w in t for w in ["cannot", "not working", "slow", "error", "bad", "fail", "problem", "broken", "issue", "bug"]):
        return "Negative"
    if any(w in t for w in ["thank", "thanks", "good", "great", "appreciate", "helpful", "well done", "love", "happy", "awesome"]):
        return "Positive"
    return "Neutral"

def local_topic(text: str) -> str:
    t = text.lower()
    if "workload" in t or "busy" in t or "task" in t:
        return "Workload"
    if "manager" in t or "support" in t or "leader" in t or "boss" in t:
        return "Management Support"
    if "office" in t or "environment" in t or "remote" in t or "home" in t:
        return "Work Environment"
    if "communication" in t or "meeting" in t or "email" in t:
        return "Communication"
    if "career" in t or "training" in t or "growth" in t or "promotion" in t:
        return "Growth"
    return "Others"

# ---------- ADAPTIVE QUESTION GENERATOR ----------
st.subheader("🧩 Generate Next Questionnaire")

if client:
    if st.button("Generate next questionnaire"):
        sentiment_summary = df["sentiment"].value_counts().to_dict()
        top_topics = df["topic"].value_counts().nlargest(3).index.tolist()
        sample_texts = "\n".join(df["message"].tail(10).tolist()) if not df.empty else "No feedback yet."

        prompt = f"""
        You are an HR assistant creating adaptive employee engagement questionnaires.

        Here are recent employee feedback comments:
        {sample_texts}

        Sentiment mix: {sentiment_summary}.
        Top themes: {', '.join(top_topics)}.

        Generate 5 short open-ended questions (under 20 words).
        Focus on areas where morale or sentiment appears negative or frustrated.
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
            else:
                st.warning("⚠️ Could not parse AI output.")
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
        dept = st.selectbox("Department", ["", "Engineering", "Finance", "HR", "Operations", "Sales", "Others"])
        answers = [st.text_area(q, height=80) for q in st.session_state.questions]
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            msg = " | ".join([a for a in answers if a.strip()])
            if msg:
                with st.spinner("🔍 Analyzing feedback..."):
                    if client:
                        try:
                            prompt = f"""
                            You are analyzing an employee feedback message.
                            Classify it into:
                            - Sentiment: [Positive, Negative, Neutral, Frustrated]
                            - Topic: [Workload, Management Support, Work Environment, Communication, Growth, Others]
                            Respond ONLY in CSV format:
                            sentiment,topic
                            Message: {msg.strip()}
                            """
                            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
                            output = resp.choices[0].message.content.strip()
                            # Skip header
                            lines = [l.strip() for l in output.splitlines() if l.strip()]
                            first = next((l for l in lines if not l.lower().startswith("sentiment")), lines[0])
                            parts = [p.strip() for p in first.split(",")]
                            raw_sent = parts[0] if len(parts) > 0 else ""
                            sentiment = normalize_sentiment(raw_sent) or local_sentiment(msg)
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
                st.success(f"✅ Feedback saved — Sentiment: **{sentiment}**, Topic: **{topic}**")
                st.rerun()

# RIGHT: QUESTION HISTORY
with right_col:
    st.subheader("📜 Questionnaire & Data Management")
    if os.path.exists("questions_history.csv"):
        hist = pd.read_csv("questions_history.csv")
        if not hist.empty:
            st.markdown("**🕓 Latest Questionnaire:**")
            last_row = hist.iloc[-1]
            st.write(f"🗓️ **Generated on:** {last_row['timestamp']}")
            for i, q in enumerate(last_row["questions"].split("|")):
                if q.strip():
                    st.markdown(f"{i+1}. {q.strip()}")
        else:
            st.caption("No questionnaires generated yet.")
    else:
        st.caption("No questionnaires found.")

    st.download_button("⬇️ Download Current Questions (CSV)",
                       data="\n".join(st.session_state.questions),
                       file_name="current_questions.csv",
                       mime="text/csv")

# ---------- DASHBOARD ----------
st.markdown("---")
st.subheader("📊 Sentiment Dashboard")

df_chart = df[df["sentiment"].isin(ALLOWED_SENTIMENTS)].copy()
if not df_chart.empty:
    sentiment_color_map = {
        "Positive": "#21bf73",
        "Negative": "#ff9f43",
        "Frustrated": "#ee5253",
        "Neutral": "#8395a7"
    }

    sent_count = df_chart["sentiment"].value_counts().reset_index()
    sent_count.columns = ["sentiment", "count"]
    fig = px.bar(sent_count, x="sentiment", y="count", color="sentiment",
                 color_discrete_map=sentiment_color_map,
                 category_orders={"sentiment": list(sentiment_color_map.keys())},
                 title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏢 Sentiment by Department")
    dept_summary = df_chart.groupby(["department", "sentiment"]).size().reset_index(name="count")
    fig2 = px.bar(dept_summary, x="department", y="count", color="sentiment",
                  barmode="group", title="Sentiment by Department",
                  color_discrete_map=sentiment_color_map)
    st.plotly_chart(fig2, use_container_width=True)

# ---------- EXECUTIVE SUMMARY ----------
st.markdown("---")
st.subheader("🧠 AI Insights Summary")

if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None

if client and st.button("Generate executive summary"):
    joined = "\n".join(df["message"].tolist())
    prompt = f"""
    Summarize HR insights:
    1) Top 3 recurring morale issues
    2) One positive highlight
    3) Mood shift trends
    4) Departments needing attention
    Keep under 250 words.
    Feedback:
    {joined}
    """
    try:
        with st.spinner("Generating executive summary..."):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            summary = resp.choices[0].message.content.strip()
            st.session_state.ai_summary = summary
            st.success("✅ New executive summary generated!")
    except Exception as e:
        st.error(f"⚠️ OpenAI error: {e}")

if st.session_state.ai_summary:
    st.markdown("### 🧾 Latest Executive Summary:")
    st.write(st.session_state.ai_summary)
else:
    st.info("No executive summary generated yet.")
