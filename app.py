import os
import random
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional import (no error if missing)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Employee Sentiment Tracker", page_icon="💬", layout="wide")
st.title("💬 Employee Sentiment Tracker")
st.caption("POC – Employee feedback sentiment analysis and adaptive survey generation")

# ---------- OPENAI CLIENT ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if (OpenAI and OPENAI_KEY) else None

# ---------- CONSTANTS ----------
DATA_FILE = "feedback.csv"
QUESTIONS_FILE = "questions_history.csv"
SUMMARY_FILE = "last_summary.txt"
TREND_FILE = "sentiment_trend.csv"

DIVISIONS = ["", "FSC", "SSO", "Crisis Shelter", "Transitional Shelter", "Care Staff", "Welfare Officer", "Others"]
ALLOWED_SENTIMENTS = ["Positive", "Negative", "Frustrated", "Neutral"]
SENTIMENT_WEIGHTS = {"Positive": 1, "Neutral": 0, "Negative": -1, "Frustrated": -2}
SENTIMENT_COLORS = {
    "Positive": "#21bf73",
    "Negative": "#ff9f43",
    "Frustrated": "#ee5253",
    "Neutral": "#8395a7",
}

# ---------- HELPERS ----------
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["id","date","employee","department","message","sentiment","topic"])

def save_data(df):
    if "id" in df.columns:
        df.drop_duplicates(subset="id", keep="last", inplace=True)
    df.to_csv(DATA_FILE, index=False)

def normalize_sentiment(val):
    if val is None or pd.isna(val): return "Neutral"
    t = str(val).strip().lower()
    if "frustrat" in t: return "Frustrated"
    if t.startswith("neg") or "bad" in t or "error" in t: return "Negative"
    if t.startswith("pos") or "good" in t or "great" in t or "happy" in t: return "Positive"
    return "Neutral"

def local_sentiment(text):
    t = (text or "").lower()
    if any(w in t for w in ["angry","frustrated","upset","annoyed","sick of","ridiculous","tired of"]): return "Frustrated"
    if any(w in t for w in ["cannot","not working","slow","error","bad","problem","fail","broken"]): return "Negative"
    if any(w in t for w in ["thank","thanks","good","great","appreciate","helpful","love","happy","awesome"]): return "Positive"
    return "Neutral"

def local_topic(text):
    t = (text or "").lower()
    if "workload" in t or "busy" in t: return "Workload"
    if "manager" in t or "support" in t: return "Management Support"
    if "office" in t or "environment" in t or "remote" in t: return "Work Environment"
    if "communication" in t or "meeting" in t: return "Communication"
    if "career" in t or "training" in t or "growth" in t: return "Growth"
    return "Others"

def classify_text_with_ai_or_local(text):
    if client is None or not text.strip():
        return local_sentiment(text), local_topic(text)
    try:
        prompt = f"""
        Classify one employee feedback message.
        Return ONLY one line: sentiment,topic
        Sentiment ∈ [Positive, Negative, Neutral, Frustrated]
        Topic ∈ [Workload, Management Support, Work Environment, Communication, Growth, Others]
        Message: {text.strip()}
        """
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
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

# ---------- SESSION STATE ----------
if "df" not in st.session_state: st.session_state.df = load_data()
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = open(SUMMARY_FILE).read().strip() if os.path.exists(SUMMARY_FILE) else None
if "questions" not in st.session_state:
    st.session_state.questions = [
        "How do you feel about your workload recently?",
        "How supported do you feel by your manager or team?",
        "Any suggestions to improve your work environment?"
    ]
df = st.session_state.df

# ---------- UPLOAD CSV ----------
st.markdown("### 📤 Upload Employee Feedback CSV")
uploaded = st.file_uploader("Upload a CSV with columns: id,date,employee,department,message", type=["csv"])

if uploaded:
    try:
        new_df = pd.read_csv(uploaded)
        new_df["message"] = new_df["message"].fillna("").astype(str)

        if client and not new_df.empty:
            messages_block = "\n".join([f"{i+1}. {m}" for i,m in enumerate(new_df["message"].tolist())])
            prompt = f"""
            Classify the following employee feedback messages into sentiment and topic.
            Return ONLY a valid CSV with columns: row_id,sentiment,topic
            Sentiment ∈ [Positive, Negative, Neutral, Frustrated]
            Topic ∈ [Workload, Management Support, Work Environment, Communication, Growth, Others]
            Messages:
            {messages_block}
            """
            with st.spinner("Classifying all feedback in one batch..."):
                resp = client.chat.completions.create(model="gpt-4o-mini",
                                                      messages=[{"role":"user","content":prompt}])
                ai_output = resp.choices[0].message.content.strip()

            import io, csv, re
            csv_lines = [l for l in ai_output.splitlines() if re.match(r"^\d+\s*,", l.strip())]
            if not any("sentiment" in l.lower() for l in csv_lines[:2]):
                csv_lines.insert(0,"row_id,sentiment,topic")
            cleaned_csv = "\n".join(csv_lines)
            try:
                result_df = pd.read_csv(io.StringIO(cleaned_csv))
            except Exception:
                reader = csv.reader(io.StringIO(cleaned_csv))
                rows=list(reader)
                header,data=rows[0],rows[1:]
                result_df=pd.DataFrame(data,columns=header)

            result_df["sentiment"]=result_df.get("sentiment","Neutral").apply(normalize_sentiment)
            new_df["sentiment"]=result_df["sentiment"]
            new_df["topic"]=result_df.get("topic","Others")

        else:
            new_df["sentiment"],new_df["topic"]=zip(*new_df["message"].apply(classify_text_with_ai_or_local))

        st.session_state.df=pd.concat([st.session_state.df,new_df],ignore_index=True)
        save_data(st.session_state.df)
        st.success(f"✅ Uploaded & classified {len(new_df)} feedback entries successfully.")
        st.rerun()

    except Exception as e:
        st.error(f"⚠️ Upload failed: {e}")

# ---------- DEMO CSV GENERATOR ----------
st.markdown("---")
if st.button("🧪 Generate Demo CSV"):
    if client:
        try:
            prompt = f"""
            Generate 10 rows of simulated employee feedback data.

            Each row = one employee responding to these {len(st.session_state.questions)} questions:
            {"; ".join(st.session_state.questions)}

            Return ONLY a CSV with columns:
            id,date,employee,department,message
            - date: today's date
            - employee: realistic first name
            - department: one of [FSC, SSO, Crisis Shelter, Transitional Shelter, Care Staff, Welfare Officer]
            - message: combine all responses separated by " | "
            Do NOT include explanations or markdown.
            """
            with st.spinner("Generating demo responses..."):
                resp = client.chat.completions.create(model="gpt-4o-mini",
                                                      messages=[{"role":"user","content":prompt}])
            raw_output=resp.choices[0].message.content.strip()

            import io, csv, re
            cleaned_output=re.sub(r"```.*?```","",raw_output,flags=re.S).strip()
            if not cleaned_output.lower().startswith("id,"):
                cleaned_output="id,date,employee,department,message\n"+cleaned_output
            try:
                demo_df=pd.read_csv(io.StringIO(cleaned_output))
            except Exception:
                reader=csv.reader(io.StringIO(cleaned_output))
                rows=[r for r in reader if len(r)>=5]
                header,data=rows[0],rows[1:]
                demo_df=pd.DataFrame(data,columns=header)

            required_cols=["id","date","employee","department","message"]
            for c in required_cols:
                if c not in demo_df.columns: demo_df[c]=""
            demo_df=demo_df[required_cols]

            if demo_df.empty:
                st.warning("⚠️ Model returned no usable data. Try again.")
            else:
                st.download_button("⬇️ Download Demo CSV",
                                   data=demo_df.to_csv(index=False).encode("utf-8"),
                                   file_name="demo_feedback.csv",
                                   mime="text/csv")
                st.success(f"✅ Generated {len(demo_df)} demo rows successfully!")
        except Exception as e:
            st.error(f"⚠️ Demo CSV generation failed: {e}")
    else:
        st.warning("No OpenAI API key detected. Please set it to enable demo CSV generation.")

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

    st.subheader("🏢 Sentiment by Division")
    dept_summary = df_live.groupby(["department", "sentiment"]).size().reset_index(name="count")
    st.plotly_chart(px.bar(dept_summary, x="department", y="count", color="sentiment",
                           barmode="group", color_discrete_map=SENTIMENT_COLORS,
                           title="Sentiment by Division"), use_container_width=True)

    if os.path.exists(TREND_FILE):
        trend_df = pd.read_csv(TREND_FILE)
        if not trend_df.empty and "avg_score" in trend_df.columns:
            st.subheader("📈 Sentiment Trend & Morale Index")
            fig1 = px.line(trend_df, x="timestamp", y="avg_score", markers=True,
                           title="Average Morale Index (Higher = Better)")
            st.plotly_chart(fig1, use_container_width=True)

# ---------- RECENT FEEDBACK ----------
st.markdown("---")
st.subheader("🗒️ Last 10 Feedback Entries")
if not df.empty:
    st.dataframe(df.sort_values("date", ascending=False).tail(10)[
        ["date", "employee", "department", "message", "sentiment", "topic"]
    ], use_container_width=True, hide_index=True)

# ---------- EXECUTIVE SUMMARY ----------
st.markdown("---")
st.subheader("🧠 Executive Summary")

if client and st.button("Generate Executive Summary"):
    joined = "\n".join(df["message"].tolist()) if not df.empty else "(no feedback)"
    trend_snippet = []
    if os.path.exists(TREND_FILE):
        tdf = pd.read_csv(TREND_FILE).tail(5)
        trend_snippet = tdf.to_dict(orient="records")

    prompt = f"""
    Summarize HR insights from employee feedback:
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
                                                  messages=[{"role":"user","content":prompt}])
            summary = resp.choices[0].message.content.strip()
            st.session_state.ai_summary = summary
            with open(SUMMARY_FILE, "w") as f:
                f.write(summary)
            update_sentiment_trend_per_run(st.session_state.df)
            st.success("✅ Summary generated and trend updated.")
            st.rerun()
    except Exception as e:
        st.error(f"⚠️ Summary generation failed: {e}")

if st.session_state.ai_summary:
    st.markdown("### 🧾 Latest Summary:")
    st.write(st.session_state.ai_summary)
