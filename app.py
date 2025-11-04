import os
import random
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Sentiment & Feedback Tracker", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background-color: #f9fafc;
        }
        h1, h2, h3 {
            color: #202A44 !important;
        }
        .stAlert {
            border-radius: 10px;
        }
        .small-note {
            color: #666;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- HEADER ----------
st.title("AI Sentiment & Feedback Tracker")
st.caption("🏆 Hackathon Proof of Concept — AI-powered employee sentiment monitoring, trend visualization & adaptive questionnaire generation")

st.info("🧭 **Flow:** ① Collect feedback → ② View sentiment trend → ③ Generate executive summaries & recommended actions → ④ Generate next questionnaire", icon="✨")

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
@st.cache_data
def load_data_cached():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=["id", "timestamp", "date", "employee", "department", "message", "sentiment", "topic"])

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

def classify_text_batch_with_ai(df):
    df = df[df["message"].notna()].copy()
    df["message"] = df["message"].astype(str)
    if df.empty:
        return ["Neutral"], ["Others"]

    msgs = "\n".join([f"{i+1}. {m}" for i, m in enumerate(df["message"].tolist()) if str(m).strip()])
    prompt = f"""
    You are classifying employee feedback entries by emotion and topic.

    Each line represents **a question and its corresponding answer**, formatted as:
    "Q: <question> | A: <answer>"

    Use the full context of both the question and the answer to determine sentiment and topic.

    For each line below, output one line in CSV format:
    sentiment,topic
    (Do not include any other row)

    ──────────────
    SENTIMENT CATEGORIES
    - Positive      = praise, satisfaction, engagement, factual improvements (“better”, “improved”, “good”)
    - Negative      = dissatisfaction or practical complaints (“reduce meetings”, “too slow”)
    - Frustrated    = emotional fatigue, stress, burnout (“tired”, “fed up”, “overwhelmed”)
    - Neutral       = purely factual statements without emotion or intent
    When mixed tones appear:
    - Determine which sentiment appears most frequently or most strongly across the answers.
    - Do not let a single disengaged or negative response dominate if most answers are neutral or constructive.
    - Only classify as Frustrated if several answers show emotional stress, exhaustion, or disengagement.

    CONTEXT AWARENESS  (important!)
    - Read both question and answer together.
    - If the question asks for positive changes or good experiences, treat factual improvements (“better communication”) as Positive.
    - If the question asks for obstacles or suggestions, treat constructive feedback (“reduce meetings”) as Negative.
    - If the answer shows disengagement or apathy (“nothing”, “don’t know”, “no teamwork”), classify as Frustrated.
    - Only use Neutral if truly emotion-free.
    - If the feedback mentions being busy, overloaded, or overwhelmed but still functional, classify as Negative instead of Frustrated.
    - If mixed tones appear, classify based on the **overall intent or energy** — if the speaker sounds motivated or engaged, mark as Positive.
    - If the topic is not explicit in the answer, infer it from the question wording.
    - If a response implies disengagement, disinterest, or absence of positivity (e.g., "nothing at all", "no teamwork"), classify as **Frustrated**.
    - Treat rhetorical or sarcastic replies as **Frustrated** or **Negative** depending on tone.

    TOPIC CATEGORIES
    [Workload, Management Support, Work Environment, Communication, Growth, Teamwork, Others]


    Feedback messages:
    {msgs}
    """
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    text = resp.choices[0].message.content.strip()
    rows = [r.strip() for r in text.split("\n") if r.strip()]
    sentiments, topics = [], []
    for line in rows:
        parts = [p.strip() for p in line.split(",")]
        sentiments.append(normalize_sentiment(parts[0]) if parts else "Neutral")
        topics.append(parts[1] if len(parts) > 1 else "Others")
    while len(sentiments) < len(df):
        sentiments.append("Neutral")
        topics.append("Others")
    return sentiments[:len(df)], topics[:len(df)]

def update_sentiment_trend_per_run(df):
    """Updates morale trend only if last update was more than 2 minutes ago."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Avoid duplicate updates within short intervals
    if os.path.exists(TREND_FILE):
        tdf = pd.read_csv(TREND_FILE)
        if not tdf.empty:
            last_time = pd.to_datetime(tdf["timestamp"].iloc[-1], errors="coerce")
            if not pd.isna(last_time):
                seconds_since_last = (datetime.now() - last_time.to_pydatetime()).total_seconds()
                if seconds_since_last < 120:
                    return  # Skip update if last entry was within 2 minutes

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
df = st.session_state.df

# ---------- TABS ----------
tabs = st.tabs(["📝 Feedback Collection", "📊 Sentiment Dashboard", "🧠 Executive Insights", "🧩 Next Questionnaire"])

# ======================================================
# TAB 1 - FEEDBACK COLLECTION
# ======================================================
with tabs[0]:
    st.subheader("📝 Employee Feedback Input")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        with st.form("form", clear_on_submit=True):
            name = st.text_input("Employee Name (optional)")
            dept = st.selectbox("Division / Department", DIVISIONS)
            answers = [st.text_area(q, height=80) for q in st.session_state.questions]
            submitted = st.form_submit_button("Submit Feedback 🚀")

            if submitted:
                msg_pairs = []
                for q, a in zip(st.session_state.questions, answers):
                    if a.strip():
                        msg_pairs.append(f"Q: {q} | A: {a.strip()}")
                msg = " || ".join(msg_pairs)
                if msg:
                    with st.spinner("🔍 Analyzing feedback with AI..."):
                        sentiment, topic = classify_text_batch_with_ai(pd.DataFrame([{"message": msg}]))
                        sentiment, topic = sentiment[0], topic[0]
                    new_row = {
                        "id": int(df["id"].max()) + 1 if len(df) else 1,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "employee": name, "department": dept,
                        "message": msg, "sentiment": sentiment, "topic": topic,
                    }
                    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
                    save_data(st.session_state.df)
                    st.success(f"✅ Saved — Sentiment: **{sentiment}**, Topic: **{topic}**")
                    update_sentiment_trend_per_run(st.session_state.df)
                    st.rerun()

    with col2:
        st.subheader("📦 Upload Batch Responses")
        st.markdown("Upload completed CSV forms (auto-analyzed by AI).")
        uploaded_q = st.file_uploader("📤 Upload CSV", type=["csv"], key="upload_questionnaire_csv_auto")
        if uploaded_q is not None:
            import hashlib
            file_bytes = uploaded_q.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            if st.session_state.get("last_uploaded_hash") == file_hash:
                st.info("✅ File already processed.")
            else:
                try:
                    df_upload = pd.read_csv(uploaded_q)
                    question_cols = [c for c in df_upload.columns if any(q[:60] in c for q in st.session_state.questions)]
                    total_rows = len(df_upload)
                    added = 0
                    current_max_id = pd.to_numeric(df["id"], errors="coerce").max()
                    if pd.isna(current_max_id): current_max_id = 0
                    next_id_start = int(current_max_id) + 1
                    progress = st.progress(0)
                    for idx, row in df_upload.iterrows():
                        employee = str(row.get("name", "")).strip()
                        dept = str(row.get("division", "")).strip()
                        msg_pairs = []
                        for q in question_cols:
                            val = str(row[q]).strip()
                            if val:
                                msg_pairs.append(f"Q: {q} | A: {val}")
                        msg = " || ".join(msg_pairs)
                        if not msg: continue
                        with st.spinner(f"Analyzing {idx + 1}/{total_rows}..."):
                            sentiment, topic = classify_text_batch_with_ai(pd.DataFrame([{"message": msg}]))
                            sentiment, topic = sentiment[0], topic[0]
                        new_row = {
                            "id": next_id_start + added,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "employee": employee,
                            "department": dept,
                            "message": msg,
                            "sentiment": sentiment,
                            "topic": topic,
                        }
                        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
                        added += 1
                        progress.progress(min((idx + 1) / max(total_rows, 1), 1.0))
                    save_data(st.session_state.df)
                    st.session_state["last_uploaded_hash"] = file_hash
                    st.success(f"✅ Uploaded & analyzed {added} feedback entr{'y' if added == 1 else 'ies'}.")
                    update_sentiment_trend_per_run(st.session_state.df)
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

# ======================================================
# TAB 2 - SENTIMENT DASHBOARD
# ======================================================
with tabs[1]:
    st.subheader("📊 Sentiment Overview")

    df_live = st.session_state.df.copy()
    df_live["sentiment"] = df_live["sentiment"].apply(normalize_sentiment)
    df_live = df_live[df_live["sentiment"].isin(ALLOWED_SENTIMENTS)]

    if not df_live.empty:
        sent_count = df_live["sentiment"].value_counts().reindex(ALLOWED_SENTIMENTS, fill_value=0).reset_index()
        sent_count.columns = ["sentiment", "count"]
        st.plotly_chart(px.bar(sent_count, x="sentiment", y="count",
                               color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                               title="Sentiment Distribution"), use_container_width=True)

        dept_summary = df_live.groupby(["department", "sentiment"]).size().reset_index(name="count")
        st.plotly_chart(px.bar(dept_summary, x="department", y="count", color="sentiment",
                               barmode="group", color_discrete_map=SENTIMENT_COLORS,
                               title="Sentiment by Division"), use_container_width=True)

        st.subheader("📈 Morale Trend Over Time")
        if os.path.exists(TREND_FILE):
            tdf = pd.read_csv(TREND_FILE)
            if not tdf.empty:
                tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
                tdf = tdf.sort_values("timestamp")
                st.plotly_chart(px.line(tdf, x="timestamp", y="avg_score", markers=True,
                                        title="Average Morale Index"), use_container_width=True)
                st.plotly_chart(px.area(tdf, x="timestamp",
                                        y=["Positive", "Negative", "Frustrated", "Neutral"],
                                        title="Sentiment Composition Over Time",
                                        color_discrete_map=SENTIMENT_COLORS),
                                use_container_width=True)

        st.markdown("### 🗒️ Recent Feedback")
        st.dataframe(df_live.sort_values("timestamp", ascending=False).head(10)[
            ["timestamp", "employee", "department", "message", "sentiment", "topic"]
        ], use_container_width=True, hide_index=True)
    else:
        st.info("No feedback yet — try submitting or uploading responses.", icon="💡")

# ======================================================
# TAB 3 - EXECUTIVE INSIGHTS
# ======================================================
with tabs[2]:
    st.subheader("🧠 Executive Summary Generator")

    if client and st.button("Generate executive summaries & recommended actions (2 formats)"):
        joined = "\n".join(df["message"].tolist()) if not df.empty else "(no feedback)"

        trend_snippet = []
        if os.path.exists(TREND_FILE):
            tdf = pd.read_csv(TREND_FILE).tail(5)
            trend_snippet = tdf.to_dict(orient="records")
        dept_list = ", ".join(df["department"].dropna().unique())

        prompt_narrative = f"""
        You are an HR communications expert summarizing employee morale based on feedback and recent trend data.

        The following morale trend data represents past sentiment summaries (newest last):
        {trend_snippet}

        Write a short HR-style narrative summary (under 250 words) that:
        - Describes how employee morale is shifting (based on the trend snippet above)
        - Highlights main pain points or frustrations
        - Notes positive or improving aspects
        - Suggests clear, people-oriented next actions for HR or management
        Include whether overall morale is improving or declining based on recent feedback.
        Avoid numeric counts and keep the tone empathetic, concise, and professional.
        Focus on qualitative patterns — e.g., “morale is improving”, “frustrations are growing”.

        Feedback:
        {joined}
        """

        prompt_bullet = f"""
        You are an HR data analyst summarizing employee feedback across divisions.

        Write a clear, reader-friendly summary (no numbers or scores).
        Focus on qualitative patterns — e.g., “morale is improving”, “frustrations are growing”.
        For each part give 1-3 bullet points and total summary need to be less than 250 words

        Include these 5 parts:
        1️ Key morale issues and frustrations.
        2️ Positive highlights or improvements.
        3️ Mood shifts over time (no numeric data), Include whether overall morale is improving or declining based on recent feedback.
        4️ Which divisions or teams seem to need more attention.
        5 Suggests and recommend clear and people-oriented next actions.

        Trend notes (for your reference): {trend_snippet}
        Divisions: {dept_list}

        Feedback data:
        {joined}
        """

        with st.spinner("Generating AI summaries..."):
            resp1 = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt_narrative}])
            resp2 = client.chat.completions.create(model="gpt-5-mini", messages=[{"role": "user", "content": prompt_bullet}])
            summary_narrative = resp1.choices[0].message.content.strip()
            summary_bullet = resp2.choices[0].message.content.strip()
            st.session_state["summary_narrative"] = summary_narrative
            st.session_state["summary_bullet"] = summary_bullet
            update_sentiment_trend_per_run(st.session_state.df)
            st.success("✅ Summaries generated & trend updated.")
            st.rerun()

    if "summary_narrative" in st.session_state or "summary_bullet" in st.session_state:
        sb, sn = st.tabs(["📝 Bullet Summary", "📋 Narrative Summary"])
        with sb:
            st.write(st.session_state.get("summary_bullet", "No summary yet."))
        with sn:
            st.write(st.session_state.get("summary_narrative", "No summary yet."))

# ======================================================
# TAB 4 - NEXT QUESTIONNAIRE
# ======================================================
with tabs[3]:
    st.subheader("🧩 Adaptive Questionnaire Builder")

    # 📈 Optional context box for HR actions/improvements
    st.markdown("#### 📈 Context for Next Round")

    # Initialize improvements state
    if "improvements" not in st.session_state:
        st.session_state.improvements = ""

    # Text area for HR to describe recent actions
    st.session_state.improvements = st.text_area(
        "Briefly describe recent improvements or management actions (optional):",
        value=st.session_state.improvements,
        placeholder="e.g. Reduced meetings, added weekly sync sessions, introduced wellness break."
    )

    # 🧠 When button is clicked
    if client and st.button("Generate next questionnaire"):
        improvements = st.session_state.improvements.strip()
        sentiment_summary = df["sentiment"].value_counts().to_dict() if not df.empty else {}
        top_topics = df["topic"].value_counts().nlargest(3).index.tolist() if not df.empty else []
        sample_texts = "\n".join(df["message"].tail(10).tolist()) if not df.empty else "No feedback yet."
        previous_qs = " | ".join(st.session_state.get("questions", []))

        # Retrieve summaries if available
        exec_summary_bullet = st.session_state.get("summary_bullet", "")
        exec_summary_narrative = st.session_state.get("summary_narrative", "")
        combined_summary = ""
        if exec_summary_bullet or exec_summary_narrative:
            combined_summary = f"""
            HR Executive Summary (AI-Generated Insights):
            - Bullet version:
            {exec_summary_bullet}

            - Narrative version:
            {exec_summary_narrative}
            """

        # Identify dominant positive and negative themes
        dominant_positive_topics, dominant_negative_topics = [], []
        if not df.empty:
            pos_df = df[df["sentiment"] == "Positive"]
            neg_df = df[df["sentiment"].isin(["Negative", "Frustrated"])]

            if not pos_df.empty:
                dominant_positive_topics = pos_df["topic"].value_counts().nlargest(2).index.tolist()
            if not neg_df.empty:
                dominant_negative_topics = neg_df["topic"].value_counts().nlargest(2).index.tolist()


        # 🧠 Build dynamic prompt
        prompt = f"""
        You are an HR assistant creating the next short employee engagement questionnaire.

        Consider the following:
        1️⃣ Recent feedback comments:
        {sample_texts}

        2️⃣ Sentiment mix: {sentiment_summary}.
        3️⃣ Top themes overall: {', '.join(top_topics)}.
        4️⃣ Positive themes to celebrate or build upon: {', '.join(dominant_positive_topics) if dominant_positive_topics else 'None detected'}.
        5️⃣ Dominant negative or frustrated areas to explore: {', '.join(dominant_negative_topics) if dominant_negative_topics else 'None detected'}.
        6️⃣ Previous questionnaire: {previous_qs}.
        7️⃣ Recent HR Executive Summary insights:
        {combined_summary}
        """

        if improvements:
            prompt += f"\n8️⃣ Recent improvements or management actions:\n{improvements}\n"

        prompt += """
        Generate 5 concise questions (<20 words each) that create a balanced mix:
        - 2 positive or uplifting questions (focus on wins, motivation, appreciation, or improvements)
        - 2 follow-up questions targeting the main negative or frustrated themes (root causes or changes)
        - 1 reflective or forward-looking question (team morale or next steps)

        Guidelines:
        - Keep tone empathetic and conversational (avoid corporate phrasing).
        - Mix open-ended and short rating/yes-no style questions.
        - Include at least one question checking whether recent improvements helped.
        - Avoid repeating or rephrasing earlier questions.
        - Number them 1–5.
        """

        # 🔮 Generate via OpenAI
        with st.spinner("Generating adaptive questionnaire with AI..."):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

        q_text = resp.choices[0].message.content
        new_qs = [
            line[line.find(".") + 1:].strip()
            for line in q_text.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]

        if new_qs:
            st.session_state.questions = new_qs
            append_questions_history(new_qs)

            st.session_state.improvements = ""
            st.session_state.improvement_box = ""  # Force UI to refresh


            st.success("✅ New questionnaire generated!")
            st.rerun()


    # 🕒 Show question history
    if os.path.exists(QUESTIONS_FILE):
        hist = pd.read_csv(QUESTIONS_FILE)
        st.markdown("### 🕒 Recent Questionnaires")
        st.dataframe(hist.tail(5).iloc[::-1], use_container_width=True, hide_index=True)

    # ⬇️ Download current questions
    st.download_button(
        "⬇️ Download Current Questions",
        "\n".join(st.session_state.questions),
        file_name="current_questions.csv",
        mime="text/csv"
    )

    # 📄 Download blank response template
    st.markdown("### 📄 Download Blank Response Template")
    if st.session_state.questions:
        headers = ["name", "division"] + [
            (q[:60] + "...") if len(q) > 60 else q for q in st.session_state.questions
        ]
        template_df = pd.DataFrame(columns=headers)
        csv_data = template_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV Template for Current Questionnaire",
            data=csv_data.encode("utf-8"),
            file_name="questionnaire_template.csv",
            mime="text/csv",
            help="Download a blank CSV with the current AI-generated questions as column headers.",
        )
