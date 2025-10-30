import streamlit as st
import pandas as pd
from datetime import datetime
import os
import plotly.express as px
import requests
from bs4 import BeautifulSoup

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="GovAI Sentiment Lens", page_icon="📊", layout="wide")
st.title("📊 GovAI Sentiment Lens")
st.caption("POC – AI-powered public feedback + live sentiment demo (Singapore context)")

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
    return pd.read_csv("feedback.csv")

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
    if any(w in t for w in ["angry", "frustrated", "upset", "annoyed", "again", "every time", "sick of", "tired of"]):
        return "Frustrated"
    if any(w in t for w in ["cannot", "not working", "slow", "error", "complain", "bad", "fail", "issue", "problem", "timeout", "crash", "inaccurate"]):
        return "Negative"
    if any(w in t for w in ["thank", "thanks", "good", "great", "appreciate", "helpful", "well done", "love"]):
        return "Positive"
    return "Neutral"

def local_topic(text: str) -> str:
    t = text.lower()
    if "singpass" in t or "login" in t or "app" in t:
        return "Digital / Singpass"
    if "bus" in t or "mrt" in t or "transport" in t:
        return "Transport"
    if "hdb" in t or "flat" in t or "rental" in t:
        return "Housing"
    if "elderly" in t or "senior" in t or "parents" in t:
        return "Elderly / Inclusion"
    if "payment" in t or "pay" in t:
        return "e-Payment"
    if "hospital" in t or "clinic" in t or "doctor" in t or "health" in t:
        return "Health"
    if "school" in t or "teacher" in t or "education" in t:
        return "Education"
    return "Others"

# ---------- ENSURE COLUMNS EXIST ----------
for col in ["sentiment", "topic"]:
    if col not in df.columns:
        df[col] = ""
missing_sent = df["sentiment"].isna() | (df["sentiment"] == "")
df.loc[missing_sent, "sentiment"] = df.loc[missing_sent, "message"].apply(local_sentiment)
missing_topic = df["topic"].isna() | (df["topic"] == "")
df.loc[missing_topic, "topic"] = df.loc[missing_topic, "message"].apply(local_topic)
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

# ---------- ADD NEW FEEDBACK ----------
st.subheader("➕ Add new citizen feedback")
with st.form("add_form", clear_on_submit=True):
    new_msg = st.text_area("Citizen feedback text")
    submitted = st.form_submit_button("Add to dashboard")
    if submitted and new_msg.strip():
        new_row = {
            "id": int(df["id"].max()) + 1 if len(df) else 1,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "message": new_msg.strip(),
            "sentiment": local_sentiment(new_msg.strip()),
            "topic": local_topic(new_msg.strip())
        }
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
        save_data(st.session_state.df)
        st.success("✅ Feedback added and saved!")

df = st.session_state.df

# ---------- SIMPLE GOOGLE SCRAPER (FALLBACK FOR DEMO) ----------
def fetch_tweets_fallback(query: str, limit: int = 5):
    """Fetch pseudo-tweets via Google search results (site:x.com Singapore)."""
    results = []
    search_query = f"site:x.com {query} singapore"
    url = f"https://www.google.com/search?q={search_query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for h in soup.select("h3")[:limit]:
            text = h.get_text().strip()
            if text:
                results.append({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "message": text,
                    "source": "Google (X.com)",
                    "user_location": "Singapore"
                })
    except Exception as e:
        st.error(f"Failed to fetch: {e}")
    return results

# ---------- UI FOR SCRAPER ----------
st.markdown("---")
st.subheader("🌐 Fetch public posts (Singapore context)")
col_a, col_b = st.columns([3, 1])
with col_a:
    twitter_query = st.text_input("Search keyword", value="singpass OR HDB OR CPF")
with col_b:
    fetch_limit = st.number_input("How many?", min_value=1, max_value=20, value=5)

if st.button("Fetch Posts (Demo)"):
    with st.spinner("Fetching simulated tweets near Singapore..."):
        tweets = fetch_tweets_fallback(twitter_query, int(fetch_limit))
        if tweets:
            st.success(f"Fetched {len(tweets)} posts")
            for t in tweets:
                st.write(f"- ({t['date']}) {t['message'][:200]} 📍 {t['user_location']}")
            new_rows = []
            start_id = int(df["id"].max()) + 1 if len(df) else 1
            for i, r in enumerate(tweets):
                new_rows.append({
                    "id": start_id + i,
                    "date": r["date"],
                    "message": r["message"],
                    "sentiment": local_sentiment(r["message"]),
                    "topic": local_topic(r["message"])
                })
            st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame(new_rows)], ignore_index=True)
            save_data(st.session_state.df)
            st.success("✅ Posts added to dataset!")
        else:
            st.warning("No results found – try another keyword like 'Singpass' or 'HDB'.")

df = st.session_state.df

# ---------- DASHBOARD ----------
st.subheader("📊 Sentiment Distribution")
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

st.subheader("🏷️ Topic Distribution")
st.bar_chart(df["topic"].value_counts())

st.subheader("📋 Feedback Table")
df_display = df.copy()
df_display.index = range(1, len(df_display) + 1)
st.dataframe(df_display[["date", "message", "sentiment", "topic"]], use_container_width=True)

# ---------- FILTER ----------
st.subheader("🔎 Filter by topic")
topic_list = ["All"] + sorted(df["topic"].unique().tolist())
selected_topic = st.selectbox("Select topic", topic_list)
if selected_topic != "All":
    df_filtered = df[df["topic"] == selected_topic].copy()
else:
    df_filtered = df.copy()
df_filtered.index = range(1, len(df_filtered) + 1)
st.dataframe(df_filtered[["date", "message", "sentiment", "topic"]])

# ---------- EXECUTIVE SUMMARY ----------
st.subheader("🧠 AI Insights Summary")
if client is None:
    st.info("Set OPENAI_API_KEY in Streamlit secrets to enable AI summary.")
else:
    if st.button("Generate executive summary"):
        joined = "\n".join(df["message"].tolist())
        prompt = f"""
        You are preparing a short insight for a GovTech / agency ops team.
        Based on the feedback below, list:
        1) Top 3 recurring citizen issues
        2) One positive highlight
        3) Any trends in frustration level or sentiment shifts
        4) Which agency or team is likely impacted
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
