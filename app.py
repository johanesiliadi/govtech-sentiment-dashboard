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

# ---------- SENTIMENT CONSTANTS ----------
ALLOWED_SENTIMENTS = ["Positive", "Negative", "Frustrated", "Neutral"]

# ---------- HELPERS ----------
@st.cache_data
def load_data():
    if os.path.exists("feedback.csv"):
        return pd.read_csv("feedback.csv")
    else:
        return pd.DataFrame(columns=["id", "date", "employee", "department", "message", "sentiment", "topic"])

def save_data(df):
    df.drop_duplicates(subset="id", keep="last", inplace=True)
    df.to_csv("feedback.csv", index=False)

def update_sentiment_trend(df):
    """Record a sentiment trend snapshot each time summary is generated."""
    trend_file = "sentiment_trend.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = df.groupby("sentiment").size().to_dict()
    for s in ALLOWED_SENTIMENTS:
        summary.setdefault(s, 0)

    new_row = {
        "timestamp": timestamp,
        "Positive": summary["Positive"],
        "Negative": summary["Negative"],
        "Frustrated": summary["Frustrated"],
        "Neutral": summary["Neutral"]
    }

    if os.path.exists(trend_file):
        trend_df = pd.read_csv(trend_file)
        trend_df = pd.concat([trend_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        trend_df = pd.DataFrame([new_row])

    trend_df.to_csv(trend_file, index=False)

# ---------- SESSION STATE ----------
if "df" not in st.session_state:
    st.session_state.df = load_data()

# ---------- LOAD LATEST QUESTIONS ----------
default_questions = [
    "How do you feel about your workload recently?",
    "How supported do you feel by your manager or team?",
    "Any suggestions to improve your work environment?"
]

if os.path.exists("questions_history.csv"):
    try:
        hist_df = pd.read_csv("questions_history.csv")
        latest_qs = [q.strip() for q in hist_df.iloc[-1]["questions"].split("|")] if not hist_df.empty else default_questions
    except Exception:
        latest_qs = default_questions
else:
    latest_qs = default_questions

if "questions" not in st.session_state:
    st.session_state.questions = latest_qs

df = st.session_state.df

# ---------- LOCAL CLASSIFIERS ----------
def local_sentiment(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["angry", "frustrated", "upset", "annoyed", "sick of", "fed up", "ridiculous", "every time", "always", "again", "tired of", "waste of time", "so bad"]):
        return "Frustrated"
    if any(w in t for w in ["cannot", "can't", "not working", "slow", "error", "bad", "fail", "problem", "broken", "issue", "bug"]):
        return "Negative"
    if any(w in t for w in ["thank", "thanks", "good", "great", "appreciate", "helpful", "well done", "love", "happy", "enjoy", "awesome", "fantastic"]):
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

# ---------- SIDEBAR LAYOUT ----------
st.markdown("---")
left_col, right_col = st.columns([1.2, 1])

# === LEFT COLUMN: Feedback Form ===
with left_col:
    st.subheader("📝 Employee Feedback Form")
    with st.form("employee_form", clear_on_submit=True):
        name = st.text_input("Employee Name (optional)")
        dept = st.selectbox("Division", ["", "FSC", "SSO", "Crisis Shelter", "Transitional Shelter", "Care Staff", "Welfare Officer", "Others"])
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
                            Classify into:
                            - Sentiment: [Positive, Negative, Neutral, Frustrated]
                            - Topic: [Workload, Management Support, Work Environment, Communication, Growth, Others]
                            Respond ONLY in CSV: sentiment,topic
                            Message: {msg.strip()}
                            """
                            resp = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": prompt}]
                            )
                            parts = [p.strip() for p in resp.choices[0].message.content.split(",")]
                            sentiment = parts[0].title() if parts else local_sentiment(msg)
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
            else:
                st.warning("Please answer at least one question.")

# === RIGHT COLUMN: Questionnaire & Data Management ===
with right_col:
    st.subheader("📜 Questionnaire & Data Management")

    if os.path.exists("questions_history.csv"):
        hist = pd.read_csv("questions_history.csv")
        if not hist.empty:
            st.markdown("**🕓 All Questionnaires:**")
            st.dataframe(hist.tail(5).iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.caption("No questionnaires generated yet.")
    else:
        st.caption("No questionnaire history available.")

    st.download_button("⬇️ Download Current Questions (CSV)",
                       data="\n".join(st.session_state.questions),
                       file_name="current_questions.csv",
                       mime="text/csv")

    if not df.empty:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download All Feedback Responses (CSV)",
                           data=csv_data,
                           file_name="employee_feedback.csv",
                           mime="text/csv")

# ---------- DASHBOARD ----------
st.markdown("---")
st.subheader("📊 Sentiment Dashboard")

if not df.empty:
    sentiment_color_map = {
        "Positive": "#21bf73",
        "Negative": "#ff9f43",
        "Frustrated": "#ee5253",
        "Neutral": "#8395a7"
    }

    # Sentiment Distribution
    sent_count = df["sentiment"].value_counts().reset_index()
    sent_count.columns = ["sentiment", "count"]
    fig = px.bar(sent_count, x="sentiment", y="count", color="sentiment",
                 color_discrete_map=sentiment_color_map,
                 category_orders={"sentiment": ALLOWED_SENTIMENTS},
                 title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment by Division
    st.subheader("🏢 Sentiment by Division")
    if "department" in df.columns and df["department"].notna().any():
        dept_summary = df.groupby(["department", "sentiment"]).size().reset_index(name="count")
        fig2 = px.bar(dept_summary, x="department", y="count", color="sentiment",
                      barmode="group", title="Sentiment by Division",
                      color_discrete_map=sentiment_color_map,
                      category_orders={"sentiment": ALLOWED_SENTIMENTS})
        st.plotly_chart(fig2, use_container_width=True)

    # Sentiment Trend Chart
    if os.path.exists("sentiment_trend.csv"):
        st.subheader("📈 Sentiment Trend (Per Summary Run)")
        trend_df = pd.read_csv("sentiment_trend.csv")
        if not trend_df.empty:
            trend_melt = trend_df.melt(id_vars="timestamp", value_vars=ALLOWED_SENTIMENTS,
                                       var_name="Sentiment", value_name="Count")
            fig3 = px.line(trend_melt, x="timestamp", y="Count", color="Sentiment",
                           markers=True, title="Sentiment Trend Over Time",
                           color_discrete_map=sentiment_color_map)
            st.plotly_chart(fig3, use_container_width=True)

# ---------- RECENT FEEDBACK ----------
st.markdown("---")
st.subheader("🗒️ Last 10 Employee Feedback Entries")

if not df.empty:
    last10 = df.sort_values(by="date", ascending=False).tail(10)
    st.dataframe(last10[["date", "employee", "department", "message", "sentiment", "topic"]],
                 use_container_width=True, hide_index=True)
else:
    st.info("No feedback available yet. Add some responses to see them here.")

# ---------- EXECUTIVE SUMMARY ----------
st.markdown("---")
st.subheader("🧠 AI Insights Summary")

if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = None

if client and st.button("Generate executive summary"):
    joined = "\n".join(df["message"].tolist())
    trend_snippet = ""
    if os.path.exists("sentiment_trend.csv"):
        tdf = pd.read_csv("sentiment_trend.csv").tail(5)
        trend_snippet = tdf.to_dict(orient="records")

    prompt = f"""
    Summarize HR insights:
    1) Top 3 recurring morale issues
    2) One positive highlight
    3) Mood shift trends (based on trend data)
    4) Divisions needing attention
    Recent sentiment trend data: {trend_snippet}
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
            update_sentiment_trend(st.session_state.df)  # record a snapshot for trend
    except Exception as e:
        st.error(f"⚠️ OpenAI error: {e}")

if st.session_state.ai_summary:
    st.markdown("### 🧾 Latest Executive Summary:")
    st.write(st.session_state.ai_summary)
else:
    st.info("No executive summary generated yet.")
