import os
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
from io import StringIO

# ---------- Optional OpenAI import ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- Page setup ----------
st.set_page_config(page_title="Employee Sentiment Tracker", page_icon="💬", layout="wide")
st.title("💬 Employee Sentiment Tracker")
st.caption("POC – AI-powered employee feedback analysis & adaptive survey generation")

# ---------- Initialize OpenAI client ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if (OpenAI and OPENAI_KEY) else None

# ---------- Constants ----------
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

# ---------- Helper functions ----------
@st.cache_data
def load_data_cached():
    """Load existing feedback data if present."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["id", "date", "employee", "department", "message", "sentiment", "topic"])

def save_data(df):
    """Save updated feedback to CSV."""
    if "id" in df.columns:
        df.drop_duplicates(subset="id", keep="last", inplace=True)
    df.to_csv(DATA_FILE, index=False)

def normalize_sentiment(val):
    """Normalize various sentiment strings into standard categories."""
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
    """Simple fallback sentiment detection."""
    t = (text or "").lower()
    if any(w in t for w in ["angry", "frustrated", "upset", "annoyed", "sick of", "ridiculous", "tired of"]):
        return "Frustrated"
    if any(w in t for w in ["cannot", "not working", "slow", "error", "bad", "problem", "fail", "broken"]):
        return "Negative"
    if any(w in t for w in ["thank", "thanks", "good", "great", "appreciate", "helpful", "love", "happy", "awesome"]):
        return "Positive"
    return "Neutral"

def local_topic(text):
    """Rule-based fallback for topic classification."""
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
    """Use AI if available, otherwise fallback to local logic."""
    if client is None or not str(text).strip():
        return local_sentiment(text), local_topic(text)
    try:
        prompt = f"""
        You are classifying an employee feedback message.
        Output ONLY one line: sentiment,topic
        Sentiment ∈ [Positive, Negative, Neutral, Frustrated]
        Topic ∈ [Workload, Management Support, Work Environment, Communication, Growth, Others]
        Message: {text.strip()}
        """
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        output = resp.choices[0].message.content.strip()
        parts = [p.strip() for p in output.split(",")]
        sentiment = normalize_sentiment(parts[0]) if parts else local_sentiment(text)
        topic = parts[1] if len(parts) > 1 else local_topic(text)
        return sentiment, topic
    except Exception:
        return local_sentiment(text), local_topic(text)

def update_sentiment_trend_per_run(df):
    """Compute morale index and save sentiment trends."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    counts = df["sentiment"].value_counts().to_dict()
    total = sum(counts.values()) or 1
    weighted_sum = sum(SENTIMENT_WEIGHTS.get(k, 0) * v for k, v in counts.items())
    avg_score = weighted_sum / total
    row = {"timestamp": timestamp, "avg_score": avg_score, **{s: counts.get(s, 0) for s in ALLOWED_SENTIMENTS}}
    tdf = pd.concat([pd.read_csv(TREND_FILE), pd.DataFrame([row])], ignore_index=True) if os.path.exists(TREND_FILE) else pd.DataFrame([row])
    tdf.to_csv(TREND_FILE, index=False)

def load_latest_questions():
    """Load last questionnaire or fallback to default."""
    default_q = [
        "How do you feel about your workload recently?",
        "How supported do you feel by your manager or team?",
        "Any suggestions to improve your work environment?",
    ]
    if not os.path.exists(QUESTIONS_FILE):
        return default_q
    try:
        df = pd.read_csv(QUESTIONS_FILE)
        if df.empty:
            return default_q
        last = df.iloc[-1]["questions"]
        return [q.strip() for q in str(last).split("|") if q.strip()]
    except Exception:
        return default_q

def append_questions_history(questions):
    """Save new questionnaire to CSV history."""
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "questions": " | ".join(questions)}
    df = pd.concat([pd.read_csv(QUESTIONS_FILE), pd.DataFrame([row])], ignore_index=True) if os.path.exists(QUESTIONS_FILE) else pd.DataFrame([row])
    df.to_csv(QUESTIONS_FILE, index=False)

# ---------- Session state ----------
if "df" not in st.session_state:
    st.session_state.df = load_data_cached()
if "questions" not in st.session_state:
    st.session_state.questions = load_latest_questions()
if "ai_summary" not in st.session_state:
    st.session_state.ai_summary = open(SUMMARY_FILE).read().strip() if os.path.exists(SUMMARY_FILE) else None
df = st.session_state.df

# ---------- Questionnaire generator ----------
st.subheader("🧩 Generate Next Questionnaire")
if client and st.button("Generate next questionnaire"):
    sentiment_summary = df["sentiment"].value_counts().to_dict() if not df.empty else {}
    top_topics = df["topic"].value_counts().nlargest(3).index.tolist() if not df.empty else []
    sample_texts = "\n".join(df["message"].tail(10).tolist()) if not df.empty else "No feedback yet."
    prompt = f"""
    You are an HR assistant creating balanced employee engagement questionnaires.
    Based on these feedback trends:
    {sample_texts}

    Sentiment mix: {sentiment_summary}
    Top themes: {', '.join(top_topics)}

    Generate 5 short open-ended questions:
    - 2 positive/reflective
    - 2 improvement-oriented
    - 1 neutral morale question
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        q_text = resp.choices[0].message.content
        new_qs = [line[line.find(".")+1:].strip() for line in q_text.splitlines() if line.strip() and line.strip()[0].isdigit()]
        if new_qs:
            st.session_state.questions = new_qs
            append_questions_history(new_qs)
            st.success("✅ Balanced questionnaire generated and saved!")
            st.rerun()
    except Exception as e:
        st.error(f"⚠️ OpenAI error: {e}")

# ---------- Layout ----------
st.markdown("---")
left, right = st.columns([1.2, 1])

# ---------- Left: Feedback Form ----------
with left:
    st.subheader("📝 Employee Feedback Form")
    with st.form("form", clear_on_submit=True):
        name = st.text_input("Employee Name (optional)")
        dept = st.selectbox("Division / Department", DIVISIONS)
        answers = [st.text_area(q, height=80) for q in st.session_state.questions]
        if st.form_submit_button("Submit Feedback"):
            msg = " | ".join([a for a in answers if a.strip()])
            if msg:
                with st.spinner("Analyzing feedback..."):
                    sentiment, topic = classify_text_with_ai_or_local(msg)
                new_row = {
                    "id": int(df["id"].max()) + 1 if len(df) else 1,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "employee": name,
                    "department": dept,
                    "message": msg,
                    "sentiment": sentiment,
                    "topic": topic,
                }
                st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
                save_data(st.session_state.df)
                st.success(f"✅ Saved — Sentiment: {sentiment}, Topic: {topic}")
                st.rerun()

# ---------- Right: Questionnaire & Data Management ----------
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

        # ---- Upload CSV & Generate Demo Data ----
    st.markdown("### 📤 Upload Employee Feedback CSV")

    uploaded = st.file_uploader(
        "Upload a CSV with columns: id,date,employee,department,message",
        type=["csv"]
    )

    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            new_df["message"] = new_df["message"].fillna("").astype(str)

            if client and not new_df.empty:
                # 🧠 Batch classify all messages in one API call
                messages_block = "\n".join(
                    [f"{i+1}. {m}" for i, m in enumerate(new_df["message"].tolist())]
                )

                prompt = f"""
                Classify the following employee feedback messages into sentiment and topic.

                Return ONLY a valid CSV with columns:
                row_id,sentiment,topic

                Sentiment ∈ [Positive, Negative, Neutral, Frustrated]
                Topic ∈ [Workload, Management Support, Work Environment, Communication, Growth, Others]

                Messages:
                {messages_block}
                """

                with st.spinner("Classifying all feedback in one batch..."):
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    ai_output = resp.choices[0].message.content.strip()

                import io, re, csv
                # --- Extract proper CSV-like lines only ---
                csv_lines = [
                    l for l in ai_output.splitlines()
                    if re.match(r"^\d+\s*,", l.strip())
                ]

                # Add header if missing
                if not any("sentiment" in l.lower() for l in csv_lines[:2]):
                    csv_lines.insert(0, "row_id,sentiment,topic")

                cleaned_csv = "\n".join(csv_lines)

                # --- Parse robustly ---
                try:
                    result_df = pd.read_csv(io.StringIO(cleaned_csv))
                except Exception:
                    reader = csv.reader(io.StringIO(cleaned_csv))
                    rows = list(reader)
                    header = rows[0]
                    data = rows[1:]
                    result_df = pd.DataFrame(data, columns=header)

                # --- Ensure required columns ---
                if "sentiment" not in result_df.columns:
                    result_df["sentiment"] = "Neutral"
                if "topic" not in result_df.columns:
                    result_df["topic"] = "Others"

                # Normalize & merge
                result_df["sentiment"] = result_df["sentiment"].apply(normalize_sentiment)
                new_df["sentiment"] = result_df["sentiment"]
                new_df["topic"] = result_df["topic"]

            else:
                # 🧩 No OpenAI key → fallback to local rule-based classification
                new_df["sentiment"], new_df["topic"] = zip(
                    *new_df["message"].apply(classify_text_with_ai_or_local)
                )

            # ✅ Save & refresh dashboard
            st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
            save_data(st.session_state.df)
            st.success(f"✅ Uploaded & classified {len(new_df)} feedback entries successfully.")
            st.rerun()

        except Exception as e:
            st.error(f"⚠️ Upload failed: {e}")

    # ---- Generate Demo CSV ----
    st.markdown("---")
    if st.button("🧪 Generate Demo CSV"):
        if client:
            try:
                prompt = f"""
                You are simulating 10 employee feedback submissions for a morale survey.

                For each simulated employee, answer all {len(st.session_state.questions)} of these questions:
                {"; ".join(st.session_state.questions)}

                Each row represents one employee.

                Return ONLY a clean CSV with columns:
                id,date,employee,department,message
                - date should be today
                - employee names can be random
                - department ∈ [FSC, SSO, Crisis Shelter, Transitional Shelter, Care Staff, Welfare Officer]
                - message should combine all answers separated by " | "
                Do not include any explanation or extra text.
                """

                with st.spinner("Generating demo responses..."):
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )
                raw_output = resp.choices[0].message.content.strip()

                import io, csv, re
                # --- Clean possible markdown or commentary ---
                lines = [l for l in raw_output.splitlines() if re.search(r"\d", l) and "," in l]
                if not any("id" in l.lower() for l in lines[:2]):
                    lines.insert(0, "id,date,employee,department,message")
                cleaned_csv = "\n".join(lines)

                # --- Parse robustly ---
                try:
                    demo_df = pd.read_csv(io.StringIO(cleaned_csv))
                except Exception:
                    reader = csv.reader(io.StringIO(cleaned_csv))
                    rows = list(reader)
                    header = rows[0]
                    data = rows[1:]
                    demo_df = pd.DataFrame(data, columns=header)

                # ✅ Guarantee correct structure
                required_cols = ["id", "date", "employee", "department", "message"]
                for col in required_cols:
                    if col not in demo_df.columns:
                        demo_df[col] = ""
                demo_df = demo_df[required_cols]

                st.download_button(
                    "⬇️ Download Demo CSV",
                    data=demo_df.to_csv(index=False).encode("utf-8"),
                    file_name="demo_feedback.csv",
                    mime="text/csv"
                )
                st.success("✅ Demo CSV generated successfully!")

            except Exception as e:
                st.error(f"⚠️ Demo CSV generation failed: {e}")
        else:
            st.warning("No OpenAI API key detected. Please set it to enable demo CSV generation.")

# ---------- Dashboard ----------
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

    # Trend chart
    if os.path.exists(TREND_FILE):
        try:
            trend_df = pd.read_csv(TREND_FILE)
            if not trend_df.empty and "avg_score" in trend_df.columns:
                st.subheader("📈 Sentiment Trend & Morale Index")
                melt = trend_df.melt(id_vars="timestamp",
                                     value_vars=[s for s in ALLOWED_SENTIMENTS if s in trend_df.columns],
                                     var_name="Sentiment", value_name="Count")
                if not melt.empty:
                    st.plotly_chart(px.line(melt, x="timestamp", y="Count", color="Sentiment",
                                            markers=True, color_discrete_map=SENTIMENT_COLORS,
                                            title="Sentiment Trend (Per Summary Run)"), use_container_width=True)
                morale_df = trend_df[["timestamp", "avg_score"]].dropna()
                if not morale_df.empty:
                    st.plotly_chart(px.line(morale_df, x="timestamp", y="avg_score", markers=True,
                                            title="Average Morale Index (Higher = Better)"), use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not load trend data: {e}")

# ---------- Recent feedback ----------
st.markdown("---")
st.subheader("🗒️ Last 10 Feedback Entries")
if not df.empty:
    st.dataframe(df.sort_values("date", ascending=False).tail(10)[
        ["date", "employee", "department", "message", "sentiment", "topic"]
    ], use_container_width=True, hide_index=True)

# ---------- Executive summary ----------
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
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
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
