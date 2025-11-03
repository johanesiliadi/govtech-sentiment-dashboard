import streamlit as st
import pandas as pd
from datetime import datetime
import os
import plotly.express as px
from openai import OpenAI

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Employee Sentiment Tracker", page_icon="💬", layout="wide")
st.title("💬 Employee Sentiment Tracker")
st.caption("POC – AI-powered analysis of employee feedback and adaptive survey generation")

# ---------- OPENAI CLIENT ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# ---------- HELPERS ----------
@st.cache_data
def load_data():
    if os.path.exists("feedback.csv"):
        return pd.read_csv("feedback.csv")
    else:
        return pd.DataFrame(columns=["id","date","employee","department","message","sentiment","topic"])

def save_data(df):
    df.drop_duplicates(subset="id", keep="last", inplace=True)
    df.to_csv("feedback.csv", index=False)

# ---------- SESSION STATE ----------
if "df" not in st.session_state:
    st.session_state.df = load_data()
if "questions" not in st.session_state:
    st.session_state.questions = [
        "How do you feel about your workload recently?",
        "How supported do you feel by your manager or team?",
        "Any suggestions to improve your work environment?"
    ]
df = st.session_state.df

# ---------- LOCAL CLASSIFIERS ----------
def local_sentiment(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["angry","frustrated","upset","annoyed","overwhelmed","stressed","tired","burnout"]):
        return "Frustrated"
    if any(w in t for w in ["cannot","not working","slow","error","bad","fail","issue","problem","negative","hard","unfair"]):
        return "Negative"
    if any(w in t for w in ["thank","thanks","good","great","appreciate","helpful","well done","love","happy","enjoy"]):
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

# ---------- EMPLOYEE FEEDBACK FORM ----------
st.subheader("📝 Employee Feedback Form")

with st.form("employee_form", clear_on_submit=True):
    name = st.text_input("Employee Name (optional)")
    dept = st.selectbox("Department", ["","Engineering","Finance","HR","Operations","Sales","Others"])
    answers = []
    for q in st.session_state.questions:
        answers.append(st.text_area(q))
    submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        msg = " | ".join([a for a in answers if a.strip()])
        if msg:
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
        else:
            st.warning("Please answer at least one question.")

df = st.session_state.df

# ---------- AI BATCH CLASSIFICATION ----------
st.subheader("🤖 AI Batch Classification (sentiment + topic)")
if st.button("Run AI Batch Classification") and client:
    prompt = """
    You are analyzing employee feedback for an organization.
    For each feedback line, classify into:
    - Sentiment: [Positive, Negative, Neutral, Frustrated]
    - Topic: [Workload, Management Support, Work Environment, Communication, Growth, Others]
    Respond ONLY in CSV format: id,sentiment,topic
    """
    for i, msg in enumerate(df["message"], start=1):
        prompt += f"{i}. {msg}\n"

    try:
        resp = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{"role": "user", "content": prompt}])
        output = resp.choices[0].message.content.strip()
        lines = [l for l in output.splitlines() if "," in l]
        parsed = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts)>=3 and parts[0].isdigit():
                parsed.append({"id": int(parts[0]), "sentiment": parts[1].title(), "topic": parts[2]})
        if parsed:
            parsed_df = pd.DataFrame(parsed)
            df = pd.merge(df, parsed_df, on="id", how="left", suffixes=("", "_ai"))
            df["sentiment"] = df["sentiment_ai"].combine_first(df["sentiment"])
            df["topic"] = df["topic_ai"].combine_first(df["topic"])
            df.drop(columns=["sentiment_ai","topic_ai"], inplace=True)
            st.session_state.df = df
            save_data(df)
            st.success(f"✅ Updated {len(parsed_df)} rows from AI output and saved.")
        else:
            st.warning("⚠️ AI returned no valid lines.")
    except Exception as e:
        st.error(f"⚠️ OpenAI error: {e}")

# ---------- DASHBOARD ----------
st.subheader("📊 Sentiment Dashboard")
if not df.empty:
    sent_count = df["sentiment"].value_counts().reset_index()
    sent_count.columns = ["sentiment","count"]
    fig = px.bar(sent_count, x="sentiment", y="count", color="sentiment",
                 color_discrete_map={"Frustrated":"red","Negative":"orange","Positive":"green","Neutral":"gray"},
                 title="Sentiment Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.bar_chart(df["topic"].value_counts(), use_container_width=True)

# ---------- ADAPTIVE QUESTION GENERATOR ----------
st.subheader("🧩 Generate Next Questionnaire")
if client:
    if st.button("Generate next questionnaire"):
        sentiment_summary = df["sentiment"].value_counts().to_dict()
        top_topics = df["topic"].value_counts().nlargest(3).index.tolist()
        summary_text = f"Sentiment mix: {sentiment_summary}. Top themes: {', '.join(top_topics)}."
        prompt = f"""
        You are an HR assistant creating adaptive employee engagement questionnaires.
        Based on these results: {summary_text}
        Generate 5 short open-ended questions (under 20 words).
        Focus on weak or frustrated areas; include one positive reflection.
        Number them 1–5.
        """
        try:
            resp = client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=[{"role": "user", "content": prompt}])
            q_text = resp.choices[0].message.content
            st.markdown("### 🆕 New Suggested Questions:")
            st.write(q_text)

            # extract numbered lines to update form
            new_qs = [line[line.find(".")+1:].strip()
                      for line in q_text.splitlines() if line.strip()[:1].isdigit()]
            if new_qs:
                st.session_state.questions = new_qs
                st.success("✅ Questionnaire updated! Refresh the page to see new questions.")
            else:
                st.warning("⚠️ Could not parse new questions.")
        except Exception as e:
            st.error(f"⚠️ OpenAI error: {e}")
else:
    st.info("Set OPENAI_API_KEY to enable adaptive question generation.")

# ---------- EXECUTIVE SUMMARY ----------
st.subheader("🧠 AI Insights Summary")
if client and st.button("Generate executive summary"):
    joined = "\n".join(df["message"].tolist())
    prompt = f"""
    Summarize HR insights:
    1) Top 3 recurring morale issues
    2) One positive highlight
    3) Mood shift trends
    4) Depts needing attention
    Keep under 150 words.
    Feedback:
    {joined}
    """
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{"role": "user", "content": prompt}])
        st.write(resp.choices[0].message.content)
    except Exception as e:
        st.error(f"⚠️ OpenAI error: {e}")
