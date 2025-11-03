# ========== Employee Sentiment Tracker ( Hackathon) ==========
# - AI questionnaire generator (history kept)
# - Manual form + auto AI classification
# - Bulk CSV upload + auto AI classification
# - Demo CSV generator (download-only; does NOT inject into dashboard)
# - Executive summary using last 5 trend snapshots (Option B: per-click)
# - Sentiment trend chart (one point per "Generate executive summary" click)
# - Consistent colors & "Last 10" table
# - Initial data preload; summary persistence to file
# ====================================================================

import os
import random
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# OpenAI is optional; app falls back to rule-based classification
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # noqa

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Employee Sentiment Tracker", page_icon="💬", layout="wide")
st.title("💬 Employee Sentiment Tracker")
st.caption("POC – AI-powered employee feedback analysis & adaptive survey generation")

# ---------- OPENAI CLIENT (optional) ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if (OpenAI and OPENAI_KEY) else None

# ---------- CONSTANTS ----------
ALLOWED_SENTIMENTS = ["Positive", "Negative", "Frustrated", "Neutral"]
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

# ========== HELPERS ==========

def ensure_initial_data() -> pd.DataFrame:
    """
    Load feedback CSV if present; otherwise create initial rows so the dashboard isn't empty.
    """
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)

    data = [
        [1, "2025-11-01", "John Tan", "FSC",
         "System keeps freezing when updating client info", "Negative", "Workload"],
        [2, "2025-11-01", "Maria Lim", "SSO",
         "Appreciate the flexible schedule recently", "Positive", "Work Environment"],
        [3, "2025-11-02", "David Lee", "Welfare Officer",
         "Sometimes it's hard to get management approval for urgent requests", "Frustrated", "Management Support"],
        [4, "2025-11-03", "Siti Rahman", "Crisis Shelter",
         "The new workflow form is confusing and slow", "Negative", "Communication"],
    ]
    df = pd.DataFrame(
        data,
        columns=["id", "date", "employee", "department", "message", "sentiment", "topic"],
    )
    df.to_csv(DATA_FILE, index=False)
    return df


@st.cache_data
def load_data_cached() -> pd.DataFrame:
    """
    Cached loader for the main data file; first call also ensures initial seed.
    """
    return ensure_initial_data()


def save_data(df: pd.DataFrame) -> None:
    """
    Save data to CSV with simple de-duplication by 'id'.
    """
    if "id" in df.columns:
        df.drop_duplicates(subset="id", keep="last", inplace=True)
    df.to_csv(DATA_FILE, index=False)


def normalize_sentiment(val: str) -> str:
    """
    Normalize free-text model outputs to one of the allowed sentiment labels.
    Prevents rogue values like "'''negative".
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "Neutral"
    t = str(val).strip().lower()

    # Heuristic mapping
    if "frustrat" in t or t in {"angry", "annoyed", "rage"}:
        return "Frustrated"
    if t.startswith("neg") or "issue" in t or "bad" in t or "error" in t:
        return "Negative"
    if t.startswith("pos") or t in {"good", "great", "happy", "awesome", "excellent"}:
        return "Positive"
    if "neutral" in t:
        return "Neutral"

    # Title-case fallback if it's already valid
    cap = t.title()
    return cap if cap in ALLOWED_SENTIMENTS else "Neutral"


def local_sentiment(text: str) -> str:
    """
    Fallback rule-based sentiment classification.
    """
    t = (text or "").lower()
    if any(w in t for w in [
        "angry", "frustrated", "upset", "annoyed", "sick of", "fed up", "ridiculous",
        "every time", "always", "again", "tired of", "waste of time", "so bad"
    ]):
        return "Frustrated"
    if any(w in t for w in [
        "cannot", "can't", "not working", "slow", "error", "bad", "fail", "broken", "issue", "bug", "problem"
    ]):
        return "Negative"
    if any(w in t for w in [
        "thank", "thanks", "good", "great", "appreciate", "helpful", "well done",
        "love", "happy", "enjoy", "awesome", "fantastic"
    ]):
        return "Positive"
    return "Neutral"


def local_topic(text: str) -> str:
    """
    Fallback rule-based topic classification.
    """
    t = (text or "").lower()
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


def classify_text_with_ai_or_local(text: str) -> tuple[str, str]:
    """
    Classify a feedback message with OpenAI if available; otherwise fallback to local rules.
    Returns (sentiment, topic).
    """
    if client is None or not text.strip():
        return local_sentiment(text), local_topic(text)

    try:
        prompt = f"""
        You are classifying an employee feedback message.
        Output ONLY a single CSV line with two fields:
        sentiment,topic

        Sentiment MUST be one of: Positive, Negative, Neutral, Frustrated
        Topic MUST be one of: Workload, Management Support, Work Environment, Communication, Growth, Others

        Message: {text.strip()}
        """
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        output = resp.choices[0].message.content.strip()
        parts = [p.strip() for p in output.split(",")]
        sentiment = normalize_sentiment(parts[0]) if len(parts) > 0 else local_sentiment(text)
        topic = parts[1] if len(parts) > 1 else local_topic(text)
        return sentiment, topic
    except Exception:
        return local_sentiment(text), local_topic(text)


def update_sentiment_trend_per_run(df: pd.DataFrame) -> None:
    """
    Option B: append a new trend snapshot on each "Generate executive summary" click.
    Snapshot includes timestamp + counts of each sentiment at that moment.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    counts = (df.groupby("sentiment").size() if "sentiment" in df.columns else pd.Series())
    summary = {s: int(counts.get(s, 0)) for s in ALLOWED_SENTIMENTS}
    new_row = {"timestamp": timestamp, **summary}

    if os.path.exists(TREND_FILE):
        tdf = pd.read_csv(TREND_FILE)
        tdf = pd.concat([tdf, pd.DataFrame([new_row])], ignore_index=True)
    else:
        tdf = pd.DataFrame([new_row])

    tdf.to_csv(TREND_FILE, index=False)


def load_latest_questions_from_history() -> list[str]:
    """
    Load the latest questionnaire version from QUESTIONS_FILE; else return defaults.
    """
    default_questions = [
        "How do you feel about your workload recently?",
        "How supported do you feel by your manager or team?",
        "Any suggestions to improve your work environment?",
    ]
    if not os.path.exists(QUESTIONS_FILE):
        return default_questions
    try:
        qdf = pd.read_csv(QUESTIONS_FILE)
        if qdf.empty:
            return default_questions
        last = qdf.iloc[-1]["questions"]
        return [q.strip() for q in str(last).split("|") if q.strip()]
    except Exception:
        return default_questions


def append_questions_history(questions: list[str]) -> None:
    """
    Add a new questionnaire version to QUESTIONS_FILE with timestamp.
    """
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "questions": " | ".join(questions),
    }
    if os.path.exists(QUESTIONS_FILE):
        qdf = pd.read_csv(QUESTIONS_FILE)
        qdf = pd.concat([qdf, pd.DataFrame([row])], ignore_index=True)
    else:
        qdf = pd.DataFrame([row])
    qdf.to_csv(QUESTIONS_FILE, index=False)


# ========== SESSION STATE BOOTSTRAP ==========

if "df" not in st.session_state:
    st.session_state.df = load_data_cached()

if "ai_summary" not in st.session_state:
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r") as f:
            st.session_state.ai_summary = f.read().strip()
    else:
        st.session_state.ai_summary = None

if "questions" not in st.session_state:
    st.session_state.questions = load_latest_questions_from_history()

df = st.session_state.df  # shorthand

# ========== ADAPTIVE QUESTION GENERATOR ==========

st.subheader("🧩 Generate Next Questionnaire")
if client and st.button("Generate next questionnaire"):
    # Build context for the model
    sentiment_summary = (df["sentiment"].value_counts().to_dict() if not df.empty else {})
    top_topics = (df["topic"].value_counts().nlargest(3).index.tolist() if not df.empty else [])
    sample_texts = "\n".join(df["message"].tail(10).tolist()) if not df.empty else "No feedback yet."

    prompt = f"""
    You are an HR assistant creating adaptive employee engagement questionnaires.

    Here are recent employee feedback comments:
    {sample_texts}

    Sentiment mix: {sentiment_summary}.
    Top themes: {', '.join(top_topics)}.

    Generate 5 short open-ended questions (under 20 words).
    Focus on areas where morale appears low (Negative or Frustrated).
    Include one positive reflection question.
    Respond as a numbered list 1-5.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        q_text = resp.choices[0].message.content
        # Parse numbered list into a clean list of questions
        new_qs = [
            line[line.find(".") + 1:].strip()
            for line in q_text.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        if new_qs:
            st.session_state.questions = new_qs
            append_questions_history(new_qs)
            st.success("✅ New questionnaire generated and saved to history!")
            st.rerun()
        else:
            st.warning("⚠️ Could not parse questions from the model output.")
    except Exception as e:
        st.error(f"⚠️ OpenAI error: {e}")
elif not client:
    st.info("Set OPENAI_API_KEY to enable adaptive question generation.")

# ========== MAIN LAYOUT ==========
st.markdown("---")
left_col, right_col = st.columns([1.2, 1])

# ---------- LEFT: FEEDBACK FORM ----------
with left_col:
    st.subheader("📝 Employee Feedback Form")
    with st.form("employee_form", clear_on_submit=True):
        name = st.text_input("Employee Name (optional)")
        dept = st.selectbox("Division / Department", DIVISIONS)
        answers = [st.text_area(q, height=80) for q in st.session_state.questions]
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            msg = " | ".join([a for a in answers if a.strip()])
            if not msg:
                st.warning("Please answer at least one question.")
            else:
                with st.spinner("🔍 Analyzing feedback..."):
                    sentiment, topic = classify_text_with_ai_or_local(msg)

                new_row = {
                    "id": int(df["id"].max()) + 1 if len(df) else 1,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "employee": name,
                    "department": dept,
                    "message": msg.strip(),
                    "sentiment": sentiment,
                    "topic": topic,
                }
                st.session_state.df = pd.concat(
                    [st.session_state.df, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                save_data(st.session_state.df)
                st.success(f"✅ Saved — Sentiment: **{sentiment}**, Topic: **{topic}**")
                st.rerun()

# ---------- RIGHT: HISTORY + DOWNLOADS + UPLOAD + DEMO CSV ----------
with right_col:
    st.subheader("📜 Questionnaire & Data Management")

    # Show full questionnaire history (latest first)
    if os.path.exists(QUESTIONS_FILE):
        q_hist = pd.read_csv(QUESTIONS_FILE)
        if not q_hist.empty:
            st.markdown("**🕓 Questionnaire History (latest first):**")
            display = q_hist.iloc[::-1].copy()
            display.index = range(1, len(display) + 1)
            display.rename(columns={"timestamp": "Generated At", "questions": "Questions"}, inplace=True)
            st.dataframe(display, use_container_width=True)
        else:
            st.caption("No questionnaires generated yet.")
    else:
        st.caption("No questionnaire history yet.")

    # Download current questions (as a 1-column CSV)
    st.download_button(
        "⬇️ Download Current Questions (CSV)",
        data="\n".join(st.session_state.questions),
        file_name="current_questions.csv",
        mime="text/csv",
    )

    # Download all feedback (for auditing/demo)
    if not df.empty:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download All Feedback Responses (CSV)",
            data=csv_data,
            file_name="employee_feedback.csv",
            mime="text/csv",
        )

    # Upload CSV: id,date,employee,department,message
    st.markdown("### 📤 Upload Bulk Feedback CSV")
    uploaded = st.file_uploader(
        "Upload CSV (columns required: id,date,employee,department,message)", type=["csv"]
    )
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            required = {"id", "date", "employee", "department", "message"}
            if not required.issubset(new_df.columns):
                st.error(f"⚠️ Missing required columns. Required: {', '.join(sorted(required))}")
            else:
                # Classify each row (AI or local)
                new_df["sentiment"], new_df["topic"] = "", ""
                with st.spinner(f"🧠 Classifying {len(new_df)} rows..."):
                    for idx, row in new_df.iterrows():
                        s, t = classify_text_with_ai_or_local(str(row["message"]))
                        new_df.at[idx, "sentiment"] = s
                        new_df.at[idx, "topic"] = t

                # Merge into existing dataset
                combined = pd.concat([st.session_state.df, new_df], ignore_index=True)
                st.session_state.df = combined
                save_data(st.session_state.df)
                st.success(f"✅ Uploaded and analyzed {len(new_df)} new responses.")
                st.rerun()
        except Exception as e:
            st.error(f"⚠️ Error reading CSV: {e}")

    # Demo CSV generator (download-only; does NOT modify main data)
    st.markdown("### 🧪 Demo CSV Generator")
    if st.button("Generate Demo CSV"):
        demo_employees = ["John Tan", "Maria Lim", "Ahmad Yusof", "Priya Menon", "Wei Ming"]
        demo_depts = ["FSC", "SSO", "Crisis Shelter", "Transitional Shelter", "Care Staff", "Welfare Officer"]

        demo_rows = []
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
                "More opportunities for career growth would be appreciated.",
            ])
            demo_rows.append({
                "id": i + 1,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "employee": emp,
                "department": dept,
                "message": msg,
            })

        demo_df = pd.DataFrame(demo_rows)
        st.success("✅ Demo CSV generated (not inserted into dashboard).")
        st.download_button(
            "⬇️ Download Demo Feedback CSV",
            data=demo_df.to_csv(index=False).encode("utf-8"),
            file_name="demo_feedback.csv",
            mime="text/csv",
        )

# ========== DASHBOARD ==========
st.markdown("---")
st.subheader("📊 Sentiment Dashboard")

df_live = st.session_state.df.copy()
# Keep only normalized/allowed sentiments
df_live["sentiment"] = df_live["sentiment"].apply(normalize_sentiment)
df_live = df_live[df_live["sentiment"].isin(ALLOWED_SENTIMENTS)]

if not df_live.empty:
    # Sentiment distribution
    sent_count = df_live["sentiment"].value_counts().reindex(ALLOWED_SENTIMENTS, fill_value=0).reset_index()
    sent_count.columns = ["sentiment", "count"]
    fig = px.bar(
        sent_count, x="sentiment", y="count", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS, title="Sentiment Distribution",
        category_orders={"sentiment": ALLOWED_SENTIMENTS},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment by Division
    st.subheader("🏢 Sentiment by Division")
    if "department" in df_live.columns and df_live["department"].notna().any():
        dept_summary = df_live.groupby(["department", "sentiment"]).size().reset_index(name="count")
        fig2 = px.bar(
            dept_summary, x="department", y="count", color="sentiment", barmode="group",
            color_discrete_map=SENTIMENT_COLORS, title="Sentiment by Division",
            category_orders={"sentiment": ALLOWED_SENTIMENTS},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Sentiment Trend (per summary run)
    if os.path.exists(TREND_FILE):
        trend_df = pd.read_csv(TREND_FILE)
        if not trend_df.empty:
            st.subheader("📈 Sentiment Trend (Per Summary Run)")
            melt = trend_df.melt(
                id_vars="timestamp",
                value_vars=ALLOWED_SENTIMENTS,
                var_name="Sentiment",
                value_name="Count",
            )
            fig3 = px.line(
                melt, x="timestamp", y="Count", color="Sentiment",
                markers=True, color_discrete_map=SENTIMENT_COLORS,
                title="Sentiment Trend Over Time (one point per summary click)"
            )
            st.plotly_chart(fig3, use_container_width=True)

# Last 10 entries
st.markdown("---")
st.subheader("🗒️ Last 10 Employee Feedback Entries")
if not st.session_state.df.empty:
    last10 = st.session_state.df.sort_values(by="date", ascending=False).tail(10)
    st.dataframe(
        last10[["date", "employee", "department", "message", "sentiment", "topic"]],
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No feedback yet. Add responses to see them here.")

# ========== EXECUTIVE SUMMARY (Option B: also logs trend per click) ==========
st.markdown("---")
st.subheader("🧠 AI Insights Summary")

if client and st.button("Generate executive summary"):
    joined = "\n".join(st.session_state.df["message"].tolist()) if not st.session_state.df.empty else "(no feedback)"
    trend_snippet = []
    if os.path.exists(TREND_FILE):
        try:
            tdf = pd.read_csv(TREND_FILE).tail(5)
            trend_snippet = tdf.to_dict(orient="records")
        except Exception:
            trend_snippet = []

    prompt = f"""
    Summarize HR insights for leadership:
    1) Top 3 recurring morale issues
    2) One positive highlight
    3) Mood shift trends (based on trend data below)
    4) Divisions needing attention

    Trend snapshots (latest 5, one per summary run): {trend_snippet}

    Keep under 250 words. Be concrete and use the trend context.
    Feedback corpus:
    {joined}
    """
    try:
        with st.spinner("Generating executive summary..."):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            summary = resp.choices[0].message.content.strip()
            st.session_state.ai_summary = summary
            # Persist summary for refresh/redeploy resilience
            with open(SUMMARY_FILE, "w") as f:
                f.write(summary)
            # Record a new snapshot for the trend (Option B)
            update_sentiment_trend_per_run(st.session_state.df)
            st.success("✅ Summary generated and trend snapshot recorded.")
            st.rerun()
    except Exception as e:
        st.error(f"⚠️ OpenAI error: {e}")

# Always show latest summary (if any)
if st.session_state.ai_summary:
    st.markdown("### 🧾 Latest Executive Summary:")
    st.write(st.session_state.ai_summary)
else:
    st.info("No executive summary generated yet.")
