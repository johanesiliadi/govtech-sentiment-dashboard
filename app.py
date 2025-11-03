import os
import random
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional OpenAI import
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Employee Sentiment Tracker", page_icon="💬", layout="wide")
st.title("💬 Employee Sentiment Tracker")
st.caption("POC – AI-powered employee feedback analysis & adaptive survey generation")

# ---------- OPENAI CLIENT ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if (OpenAI and OPENAI_KEY) else None

# ---------- CONSTANTS ----------
ALLOWED_SENTIMENTS = ["Positive", "Negative", "Frustrated", "Neutral"]
SENTIMENT_WEIGHTS = {"Positive": 1, "Neutral": 0, "Negative": -1, "Frustrated": -2}
DIVISIONS = ["", "FSC", "SSO", "Crisis Shelter", "Transitional Shelter", "Care Staff", "Welfare Officer", "Others"]
SENTIMENT_COLORS = {
    "Positive": "#21bf73",
    "Negative": "#ff9f43",
    "Frustrated": "#ee5253",
    "Neutral":  "#8395a7",
}
DATA_FILE = "feedback.csv"
QUESTIONS_FILE = "questions_history.csv"
SUMMARY_FILE = "last_summary.txt"
TREND_FILE = "sentiment_trend.csv"

# ---------- HELPERS ----------
def ensure_initial_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    data = [
        [1, "2025-11-01", "John Tan", "FSC", "System keeps freezing when updating client info", "Negative", "Workload"],
        [2, "2025-11-01", "Maria Lim", "SSO", "Appreciate the flexible schedule recently", "Positive", "Work Environment"],
        [3, "2025-11-02", "David Lee", "Welfare Officer", "Sometimes it's hard to get management approval for urgent requests", "Frustrated", "Management Support"],
        [4, "2025-11-03", "Siti Rahman", "Crisis Shelter", "The new workflow form is confusing and slow", "Negative", "Communication"],
    ]
    df = pd.DataFrame(data, columns=["id", "date", "employee", "department", "message", "sentiment", "topic"])
    df.to_csv(DATA_FILE, index=False)
    return df

@st.cache_data
def load_data_cached():
    return ensure_initial_data()

def save_data(df):
    if "id" in df.columns:
        df.drop_duplicates(subset="id", keep="last", inplace=True)
    df.to_csv(DATA_FILE, index=False)

def normalize_sentiment(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "Neutral"
    t = str(val).strip().lower()
    if "frustrat" in t:
        return "Frustrated"
    if t.startswith("neg") or "bad" in t or "error" in t:
        return "Negative"
    if t.startswith("pos") or "good" in t or "great" in t or "happy" in t:
        return "Positive"
    if "neutral" in t:
        return "Neutral"
    return "Neutral"

def local_sentiment(text):
    t = (text or "").lower()
    if any(w in t for w in ["angry", "frustrated", "upset", "annoyed", "sick of", "ridiculous", "tired of"]):
        return "Frustrated"
    if any(w in t for w in ["cannot", "not working", "slow", "error", "bad", "problem", "fail", "broken"]):
        return "Negative"
    if any(w in t for w in ["thank", "thanks", "good", "great", "appreciate", "helpful", "love", "happy", "awesome"]):
        return "Positive"
    return "Neutral"

def local_topic(text):
    t = (text or "").lower()
    if "workload" in t or "busy" in t:
        return "Workload"
    if "manager" in t or "support" in t:
        return "Management Support"
    if "office" in t or "environment" in t:
        return "Work Environment"
    if "communication" in t or "meeting" in t:
        return "Communication"
    if "career" in t or "training" in t or "growth" in t:
        return "Growth"
    return "Others"

def classify_text_with_ai_or_local(text):
    if client is None or not text.strip():
        return local_sentiment(text), local_topic(text)
    try:
        prompt = f"""
        You are classifying an employee feedback message.
        Output ONLY one line: sentiment,topic
        Sentiment ∈ [Positive, Negative, Neutral, Frustrated]
        Topic ∈ [Workload, Management Support, Work Environment, Communication, Growth, Others]
        Message: {text.strip()}
        """
        resp = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{"role": "user", "content": prompt}])
        output = resp.choices[0].message.content.strip()
        parts = [p.strip() for p in output.split(",")]
        sentiment = normalize_sentiment(parts[0]) if parts else local_sentiment(text)
        topic = parts[1] if len(parts) > 1 else local_topic(text)
        return sentiment, topic
    except Exception:
        return local_sentiment(text), local_topic(text)

def update_sentiment_trend_per_run(df):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    counts = df["sentiment"].value_counts().to_dict()
    total = sum(counts.values()) or 1
    weighted_sum = sum(SENTIMENT_WEIGHTS.get(k, 0) * v for k, v in counts.items())
    avg_score = weighted_sum / total
    row = {"timestamp": timestamp, "avg_score": avg_score, **{s: counts.get(s, 0) for s in ALLOWED_SENTIMENTS}}
    if os.path.exists(TREND_FILE):
        tdf = pd.read_csv(TREND_FILE)
        tdf = pd.concat([tdf, pd.DataFrame([row])], ignore_index=True)
    else:
        tdf = pd.DataFrame([row])
    tdf.to_csv(TREND_FILE, index=False)

def load_latest_questions():
    default_q = [
        "How do you feel about your workload recently?",
        "How supported do you feel by your manager or team?",
        "Any suggestions to improve your work environment?",
    ]
    if not os.path.exists(QUESTIONS_FILE): return default_q
    try:
        df = pd.read_csv(QUESTIONS_FILE)
        if df.empty: return default_q
        last = df.iloc[-1]["questions"]
        return [q.strip() for q in str(last).split("|") if q.strip()]
    except Exception:
        return default_q

def append_questions_history(questions):
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "questions": " | ".join(questions)}
    if os.path.exists(QUESTIONS_FILE):
        df = pd.read_csv(QUESTIONS_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(QUESTIONS_FILE, index=False)

# ---------- STATE ----------
if "df" not in st.session_state: st.session_state.df = load_data_cached()
if "questions" not in st.session_state: st.session_state.questions = load_latest_questions()
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = open(SUMMARY_FILE).read().strip() if os.path.exists(SUMMARY_FILE) else None
df = st.session_state.df

# ---------- BALANCED QUESTION GENERATOR (Option A) ----------
st.subheader("🧩 Generate Next Questionnaire")
if client and st.button("Generate next questionnaire"):
    sentiment_summary = df["sentiment"].value_counts().to_dict() if not df.empty else {}
    top_topics = df["topic"].value_counts().nlargest(3).index.tolist() if not df.empty else []
    sample_texts = "\n".join(df["message"].tail(10).tolist()) if not df.empty else "No feedback yet."

    prompt = f"""
    You are an HR assistant creating balanced employee engagement questionnaires.
    Here are recent feedback comments:
    {sample_texts}
    Sentiment mix: {sentiment_summary}.
    Top themes: {', '.join(top_topics)}.

    Generate 5 open-ended questions (under 20 words total):
    - 2 exploring positive experiences or improvements
    - 2 addressing areas needing support or change
    - 1 neutral morale-reflection question
    Number them 1–5.
    """
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{"role": "user", "content": prompt}])
        q_text = resp.choices[0].message.content
        new_qs = [line[line.find(".")+1:].strip()
                  for line in q_text.splitlines()
                  if line.strip() and line.strip()[0].isdigit()]
        if new_qs:
            st.session_state.questions = new_qs
            append_questions_history(new_qs)
            st.success("✅ Balanced questionnaire generated and saved!")
            st.rerun()
    except Exception as e:
        st.error(f"⚠️ OpenAI error: {e}")

# ---------- LAYOUT ----------
st.markdown("---")
left, right = st.columns([1.2, 1])

# ---------- LEFT: FORM ----------
with left:
    st.subheader("📝 Employee Feedback Form")
    with st.form("form", clear_on_submit=True):
        name = st.text_input("Employee Name (optional)")
        dept = st.selectbox("Division / Department", DIVISIONS)
        answers = [st.text_area(q, height=80) for q in st.session_state.questions]
        if st.form_submit_button("Submit Feedback"):
            msg = " | ".join([a for a in answers if a.strip()])
            if msg:
                with st.spinner("🔍 Analyzing feedback..."):
                    sentiment, topic = classify_text_with_ai_or_local(msg)
                new_row = {
                    "id": int(df["id"].max()) + 1 if len(df) else 1,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "employee": name, "department": dept,
                    "message": msg, "sentiment": sentiment, "topic": topic,
                }
                st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
                save_data(st.session_state.df)
                st.success(f"✅ Saved — Sentiment: {sentiment}, Topic: {topic}")
                st.rerun()

# ---------- RIGHT: MANAGEMENT ----------
with right:
    st.subheader("📜 Questionnaire & Data Management")
    if os.path.exists(QUESTIONS_FILE):
        hist = pd.read_csv(QUESTIONS_FILE)
        if not hist.empty:
            st.dataframe(hist.tail(5).iloc[::-1], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download Current Questions",
                       data="\n".join(st.session_state.questions),
                       file_name="current_questions.csv", mime="text/csv")

    if not df.empty:
        st.download_button("⬇️ Download All Feedback",
                           data=df.to_csv(index=False).encode("utf-8"),
                           file_name="employee_feedback.csv", mime="text/csv")

    uploaded = st.file_uploader("📤 Upload CSV (id,date,employee,department,message)", type=["csv"])
    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            new_df["sentiment"], new_df["topic"] = zip(*new_df["message"].apply(classify_text_with_ai_or_local))
            st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
            save_data(st.session_state.df)
            st.success(f"✅ Uploaded & classified {len(new_df)} entries.")
            st.rerun()
        except Exception as e:
            st.error(f"Upload error: {e}")

    if st.button("🧪 Generate Demo CSV"):
        demo = [{"id": i+1, "date": datetime.now().strftime("%Y-%m-%d"),
                 "employee": random.choice(["John", "Maria", "Priya", "Wei Ming"]),
                 "department": random.choice(DIVISIONS[1:]),
                 "message": f"Feedback: {random.choice(st.session_state.questions)}"}
                for i in range(10)]
        demo_df = pd.DataFrame(demo)
        st.download_button("⬇️ Download Demo Feedback CSV",
                           data=demo_df.to_csv(index=False).encode("utf-8"),
                           file_name="demo_feedback.csv", mime="text/csv")

# ---------- DASHBOARD ----------
st.markdown("---")
st.subheader("📊 Sentiment Dashboard")

df_live = st.session_state.df.copy()
df_live["sentiment"] = df_live["sentiment"].apply(normalize_sentiment)
df_live = df_live[df_live["sentiment"].isin(ALLOWED_SENTIMENTS)]

if not df_live.empty:
    sent_count = df_live["sentiment"].value_counts().reindex(ALLOWED_SENTIMENTS, fill_value=0).reset_index()
    sent_count.columns = ["sentiment", "count"]
    st.plotly_chart(px.bar(sent_count, x="sentiment", y="count",
                           color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                           title="Sentiment Distribution"), use_container_width=True)

    if "department" in df_live.columns:
        st.subheader("🏢 Sentiment by Division")
        dept_summary = df_live.groupby(["department", "sentiment"]).size().reset_index(name="count")
        st.plotly_chart(px.bar(dept_summary, x="department", y="count", color="sentiment",
                               barmode="group", color_discrete_map=SENTIMENT_COLORS,
                               title="Sentiment by Division"), use_container_width=True)

    if os.path.exists(TREND_FILE):
        trend_df = pd.read_csv(TREND_FILE)
        if not trend_df.empty:
            st.subheader("📈 Sentiment Trend & Morale Index")
            melt = trend_df.melt(id_vars="timestamp", value_vars=ALLOWED_SENTIMENTS,
                                 var_name="Sentiment", value_name="Count")
            st.plotly_chart(px.line(melt, x="timestamp", y="Count", color="Sentiment",
                                    markers=True, color_discrete_map=SENTIMENT_COLORS,
                                    title="Sentiment Trend (Per Summary Run)"),
                            use_container_width=True)
            st.plotly_chart(px.line(trend_df, x="timestamp", y="avg_score", markers=True,
                                    title="Average Morale Index (Higher = Better)"),
                            use_container_width=True)

# ---------- RECENT FEEDBACK ----------
st.markdown("---")
st.subheader("🗒️ Last 10 Feedback Entries")
if not df.empty:
    st.dataframe(df.sort_values("date", ascending=False).tail(10)[
        ["date", "employee", "department", "message", "sentiment", "topic"]
    ], use_container_width=True, hide_index=True)

# ---------- EXECUTIVE SUMMARY ----------
st.markdown("---")
st.subheader("🧠 AI Insights Summary")

if client and st.button("Generate executive summary"):
    joined = "\n".join(df["message"].tolist()) if not df.empty else "(no feedback)"
    trend_snippet = []
    if os.path.exists(TREND_FILE):
        tdf = pd.read_csv(TREND_FILE).tail(5)
        trend_snippet = tdf.to_dict(orient="records")
    prompt = f"""
    Summarize HR insights:
    1) Top 3 morale issues
    2) One positive highlight
    3) Mood shift trends (based on morale index)
    4) Divisions needing attention
    Trend snapshots: {trend_snippet}
    Keep under 250 words.
    Feedback:
    {joined}
    """
    try:
        with st.spinner("Generating executive summary..."):
            resp = client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=[{"role": "user", "content": prompt}])
            summary = resp.choices[0].message.content.strip()
            st.session_state.ai_summary = summary
            with open(SUMMARY_FILE, "w") as f:
                f.write(summary)
            update_sentiment_trend_per_run(st.session_state.df)
            st.success("✅ Summary generated & trend updated.")
            st.rerun()
    except Exception as e:
        st.error(f"⚠️ OpenAI error: {e}")

if st.session_state.ai_summary:
    st.markdown("### 🧾 Latest Executive Summary:")
    st.write(st.session_state.ai_summary)
