import streamlit as st
import pandas as pd
from datetime import datetime
import os
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Employee Sentiment Tracker", page_icon="💬", layout="wide")
st.title("💬 Employee Sentiment Tracker")
st.caption("POC – AI-powered analysis of employee feedback and engagement sentiment")

# ---------- OPENAI CLIENT ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None

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

# ---------- SESSION STATE ----------
if "df" not in st.session_state:
    st.session_state.df = load_data()
df = st.session_state.df

# ---------- LOCAL CLASSIFIERS ----------
def local_sentiment(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["angry", "frustrated", "upset", "annoyed", "overwhelmed", "stressed", "tired", "burnout"]):
        return "Frustrated"
    if any(w in t for w in ["cannot", "not working", "slow", "error", "bad", "fail", "issue", "problem", "negative", "hard", "unfair"]):
        return "Negative"
    if any(w in t for w in ["thank", "thanks", "good", "great", "appreciate", "helpful", "well done", "love", "happy", "enjoy"]):
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

# ---------- ENSURE COLUMNS ----------
for col in ["employee", "department", "sentiment", "topic"]:
    if col not in df.columns:
        df[col] = ""
save_data(df)

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Configuration")
use_ai = st.sidebar.toggle("Enable AI mode", value=True)
if not use_ai:
    client = None
    st.sidebar.warning("Running in LOCAL MODE (rule-based only)")
else:
    if client:
        st.sidebar.success("Running in AI MODE (OpenAI active)")
    else:
        st.sidebar.error("No OpenAI key found – defaulting to LOCAL MODE")

# ---------- ADD NEW EMPLOYEE FEEDBACK ----------
st.subheader("📝 Employee Feedback Form")
with st.form("employee_form", clear_on_submit=True):
    name = st.text_input("Employee Name (optional)")
    dept = st.selectbox("Department", ["", "Engineering", "Finance", "HR", "Operations", "Sales", "Others"])
    q1 = st.text_area("How do you feel about your workload recently?")
    q2 = st.text_area("How supported do you feel by your manager or team?")
    q3 = st.text_area("Any suggestions to improve your work environment?")
    submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        msg = " | ".join([q1, q2, q3])
        new_row = {
            "id": int(df["id"].max()) + 1 if len(df) else 1,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "employee": name,
            "department": dept,
            "message": msg.strip(),
            "sentiment": local_sentiment(msg.strip()),
            "topic": local_topic(msg.strip())
        }
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
        save_data(st.session_state.df)
        st.success("✅ Feedback saved and analyzed!")

df = st.session_state.df

# ---------- AI BATCH CLASSIFICATION ----------
st.subheader("🤖 AI Batch Classification (sentiment + topic)")
st.write("Analyze all feedback using AI and update results.")

if st.button("Run AI Batch Classification"):
    if client is None:
        st.warning("⚠️ No OpenAI key found, using local classification only.")
    else:
        prompt = """
        You are analyzing employee feedback for an organization.

        For each feedback line below, classify into:
        - Sentiment: One of [Positive, Negative, Neutral, Frustrated]
        - Topic: One of [Workload, Management Support, Work Environment, Communication, Growth, Others]

        Respond ONLY in CSV format:
        id,sentiment,topic
        """
        for i, msg in enumerate(df["message"], start=1):
            prompt += f"{i}. {msg}\n"

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            output = resp.choices[0].message.content.strip()

            lines = [l.strip() for l in output.splitlines() if "," in l]
            parsed = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3 and parts[0].isdigit():
                    parsed.append({
                        "id": int(parts[0]),
                        "sentiment": parts[1].title(),
                        "topic": parts[2]
                    })

            if parsed:
                parsed_df = pd.DataFrame(parsed)
                df = pd.merge(df, parsed_df, on="id", how="left", suffixes=("", "_ai"))
                df["sentiment"] = df["sentiment_ai"].combine_first(df["sentiment"])
                df["topic"] = df["topic_ai"].combine_first(df["topic"])
                df.drop(columns=["sentiment_ai", "topic_ai"], inplace=True)
                st.session_state.df = df
                save_data(st.session_state.df)
                st.success(f"✅ Updated {len(parsed_df)} rows from AI output and saved.")
            else:
                st.warning("⚠️ AI did not return valid CSV lines.")
        except Exception as e:
            st.error(f"⚠️ OpenAI error: {e}")

# ---------- DASHBOARD ----------
st.subheader("📊 Sentiment Distribution (Employee Mood)")
if not df.empty:
    sent_count = df["sentiment"].value_counts().reset_index()
    sent_count.columns = ["sentiment", "count"]
    fig = px.bar(
        sent_count,
        x="sentiment",
        y="count",
        title="Sentiment Distribution",
        color="sentiment",
        color_discrete_map={
            "Frustrated": "red",
            "Negative": "orange",
            "Positive": "green",
            "Neutral": "gray"
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏢 Sentiment by Department")
    if "department" in df.columns and df["department"].notna().any():
        dept_summary = df.groupby(["department", "sentiment"]).size().reset_index(name="count")
        fig2 = px.bar(
            dept_summary,
            x="department",
            y="count",
            color="sentiment",
            barmode="group",
            title="Sentiment by Department"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🏷️ Topic Distribution")
    st.bar_chart(df["topic"].value_counts())

    st.subheader("📋 Feedback Table")
    df_display = df.copy()
    df_display.index = range(1, len(df_display) + 1)
    st.dataframe(df_display[["date", "employee", "department", "message", "sentiment", "topic"]], use_container_width=True)

# ---------- FILTER ----------
st.subheader("🔎 Filter by Department or Topic")
col1, col2 = st.columns(2)
dept_filter = col1.selectbox("Select Department", ["All"] + sorted(df["department"].dropna().unique().tolist()))
topic_filter = col2.selectbox("Select Topic", ["All"] + sorted(df["topic"].dropna().unique().tolist()))

df_filtered = df.copy()
if dept_filter != "All":
    df_filtered = df_filtered[df_filtered["department"] == dept_filter]
if topic_filter != "All":
    df_filtered = df_filtered[df_filtered["topic"] == topic_filter]

df_filtered.index = range(1, len(df_filtered) + 1)
st.dataframe(df_filtered[["date", "employee", "department", "message", "sentiment", "topic"]])

# ---------- EXECUTIVE SUMMARY ----------
st.subheader("🧠 AI Insights Summary")
if client is None:
    st.info("Set OPENAI_API_KEY in Streamlit secrets to enable AI summary.")
else:
    if st.button("Generate executive summary"):
        joined = "\n".join(df["message"].tolist())
        prompt = f"""
        You are preparing a short insight for HR leadership based on employee feedback.

        Summarize the following:
        1) Top 3 recurring employee issues or morale concerns
        2) One positive highlight or trend
        3) Any shifts in overall mood (frustration vs positivity)
        4) Which departments may require attention
        Keep it under 150 words.

        Feedback:
        {joined}
        """
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            st.write(resp.choices[0].message.content)
        except Exception as e:
            st.error(f"⚠️ OpenAI error: {e}")
