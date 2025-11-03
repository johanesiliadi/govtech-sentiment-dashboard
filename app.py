import os, random, re, io, csv
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

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
    "Positive": "#21bf73", "Negative": "#ff9f43", "Frustrated": "#ee5253", "Neutral": "#8395a7",
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
        resp = client.chat.completions.create(model="gpt-4o-mini",
                                              messages=[{"role":"user","content":prompt}])
        out = resp.choices[0].message.content.strip()
        parts = [p.strip() for p in out.split(",")]
        s = normalize_sentiment(parts[0]) if parts else local_sentiment(text)
        t = parts[1] if len(parts)>1 else local_topic(text)
        return s,t
    except Exception:
        return local_sentiment(text), local_topic(text)

def update_sentiment_trend_per_run(df):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    counts = df["sentiment"].value_counts().to_dict()
    total = sum(counts.values()) or 1
    score = sum(SENTIMENT_WEIGHTS.get(k,0)*v for k,v in counts.items())/total
    row = {"timestamp":ts,"avg_score":score,**{s:counts.get(s,0) for s in ALLOWED_SENTIMENTS}}
    if os.path.exists(TREND_FILE):
        tdf = pd.read_csv(TREND_FILE)
        tdf = pd.concat([tdf,pd.DataFrame([row])],ignore_index=True)
    else:
        tdf = pd.DataFrame([row])
    tdf.to_csv(TREND_FILE,index=False)

def append_questions_history(qs):
    new_row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "questions": " | ".join(qs)}
    if os.path.exists(QUESTIONS_FILE):
        hist = pd.read_csv(QUESTIONS_FILE)
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    else:
        hist = pd.DataFrame([new_row])
    hist.to_csv(QUESTIONS_FILE, index=False)

def load_question_history():
    if not os.path.exists(QUESTIONS_FILE):
        return pd.DataFrame(columns=["timestamp","questions"])
    return pd.read_csv(QUESTIONS_FILE)

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

# ---------- LAYOUT ----------
st.markdown("---")
left, right = st.columns([1.3,1])

# ---------- LEFT COLUMN : FEEDBACK FORM ----------
with left:
    st.subheader("📝 Employee Feedback Form")
    with st.form("form", clear_on_submit=True):
        name = st.text_input("Employee Name (optional)")
        dept = st.selectbox("Division / Department", DIVISIONS)
        answers = [st.text_area(q, height=70) for q in st.session_state.questions]
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            msg = " | ".join([a for a in answers if a.strip()])
            if msg:
                with st.spinner("Analyzing feedback..."):
                    s,t = classify_text_with_ai_or_local(msg)
                new_row = {
                    "id": int(df["id"].max())+1 if len(df) else 1,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "employee": name, "department": dept,
                    "message": msg, "sentiment": s, "topic": t
                }
                st.session_state.df = pd.concat([st.session_state.df,pd.DataFrame([new_row])],ignore_index=True)
                save_data(st.session_state.df)
                st.success(f"✅ Saved — Sentiment: {s}, Topic: {t}")
                st.rerun()
            else:
                st.warning("Please fill in at least one question.")

# ---------- RIGHT COLUMN : DATA MANAGEMENT ----------
with right:
    st.subheader("📂 Data Management")

    # Upload CSV
    uploaded = st.file_uploader("Upload CSV (id,date,employee,department,message)", type=["csv"])
    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            new_df["message"] = new_df["message"].fillna("").astype(str)
            if client and not new_df.empty:
                messages = "\n".join([f"{i+1}. {m}" for i,m in enumerate(new_df["message"])])
                prompt = f"""
                Classify the following feedback messages.
                Return only CSV: row_id,sentiment,topic
                {messages}
                """
                resp = client.chat.completions.create(model="gpt-4o-mini",
                                                      messages=[{"role":"user","content":prompt}])
                out = resp.choices[0].message.content.strip()
                csv_lines = [l for l in out.splitlines() if re.match(r"^\d+,",l)]
                if not any("sentiment" in l.lower() for l in csv_lines[:2]):
                    csv_lines.insert(0,"row_id,sentiment,topic")
                cleaned="\n".join(csv_lines)
                try: result = pd.read_csv(io.StringIO(cleaned))
                except: result=pd.DataFrame(csv.reader(io.StringIO(cleaned)))
                new_df["sentiment"]=result.get("sentiment","Neutral").apply(normalize_sentiment)
                new_df["topic"]=result.get("topic","Others")
            else:
                new_df["sentiment"],new_df["topic"]=zip(*new_df["message"].apply(classify_text_with_ai_or_local))
            st.session_state.df=pd.concat([st.session_state.df,new_df],ignore_index=True)
            save_data(st.session_state.df)
            st.success(f"✅ Uploaded & classified {len(new_df)} entries.")
            st.rerun()
        except Exception as e:
            st.error(f"⚠️ Upload failed: {e}")

    # Generate Demo CSV
    if st.button("🧪 Generate Demo CSV"):
        if client:
            try:
                prompt=f"""
                Generate 10 rows of employee feedback data.
                Columns: id,date,employee,department,message
                department ∈ [FSC, SSO, Crisis Shelter, Transitional Shelter, Care Staff, Welfare Officer]
                message = combined answers to: {"; ".join(st.session_state.questions)}
                """
                resp=client.chat.completions.create(model="gpt-4o-mini",
                                                    messages=[{"role":"user","content":prompt}])
                raw=resp.choices[0].message.content.strip()
                cleaned=re.sub(r"```.*?```","",raw,flags=re.S).strip()
                if not cleaned.lower().startswith("id,"):
                    cleaned="id,date,employee,department,message\n"+cleaned
                try: demo=pd.read_csv(io.StringIO(cleaned))
                except:
                    reader=csv.reader(io.StringIO(cleaned))
                    rows=[r for r in reader if len(r)>=5]; demo=pd.DataFrame(rows[1:],columns=rows[0])
                st.download_button("⬇️ Download Demo CSV",
                                   data=demo.to_csv(index=False).encode("utf-8"),
                                   file_name="demo_feedback.csv",
                                   mime="text/csv")
                st.success("✅ Demo CSV ready for download!")
            except Exception as e:
                st.error(f"⚠️ Demo generation failed: {e}")
        else:
            st.warning("OpenAI key not found.")

    # Show Question History
    hist_df = load_question_history()
    if not hist_df.empty:
        st.markdown("### 🕓 Recent Questionnaires")
        st.dataframe(hist_df.tail(3).iloc[::-1], use_container_width=True, hide_index=True)

# ---------- DASHBOARD ----------
st.markdown("---")
st.subheader("📊 Sentiment Dashboard")
df_live = st.session_state.df.copy()
df_live["sentiment"]=df_live["sentiment"].apply(normalize_sentiment)
df_live=df_live[df_live["sentiment"].isin(ALLOWED_SENTIMENTS)]

if not df_live.empty:
    sent_count=df_live["sentiment"].value_counts().reindex(ALLOWED_SENTIMENTS,fill_value=0).reset_index()
    sent_count.columns=["sentiment","count"]
    st.plotly_chart(px.bar(sent_count,x="sentiment",y="count",color="sentiment",
                           color_discrete_map=SENTIMENT_COLORS,
                           title="Sentiment Distribution"),use_container_width=True)
    dept=df_live.groupby(["department","sentiment"]).size().reset_index(name="count")
    st.plotly_chart(px.bar(dept,x="department",y="count",color="sentiment",
                           barmode="group",color_discrete_map=SENTIMENT_COLORS,
                           title="Sentiment by Division"),use_container_width=True)

# ---------- RECENT FEEDBACK ----------
st.markdown("---")
st.subheader("🗒️ Last 10 Feedback Entries")
if not df.empty:
    st.dataframe(df.sort_values("date",ascending=False).tail(10)[
        ["date","employee","department","message","sentiment","topic"]],
        use_container_width=True,hide_index=True)

# ---------- EXECUTIVE SUMMARY ----------
st.markdown("---")
st.subheader("🧠 Executive Summary")

if client and st.button("Generate Executive Summary"):
    joined="\n".join(df["message"]) if not df.empty else "(no feedback)"
    trend=pd.read_csv(TREND_FILE).tail(5).to_dict(orient="records") if os.path.exists(TREND_FILE) else []
    prompt=f"""
    Summarize HR insights:
    1) Top morale issues
    2) Positive highlights
    3) Mood shift trends
    4) Divisions needing attention
    Trend data: {trend}
    Feedback:
    {joined}
    """
    with st.spinner("Generating summary..."):
        resp=client.chat.completions.create(model="gpt-4o-mini",
                                            messages=[{"role":"user","content":prompt}])
        summary=resp.choices[0].message.content.strip()
        st.session_state.ai_summary=summary
        with open(SUMMARY_FILE,"w") as f: f.write(summary)
        update_sentiment_trend_per_run(st.session_state.df)
        st.success("✅ Summary generated & trend updated.")
        st.rerun()

if st.session_state.ai_summary:
    st.markdown("### 🧾 Latest Summary:")
    st.write(st.session_state.ai_summary)
