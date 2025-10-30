import streamlit as st
import pandas as pd
from datetime import datetime
import os
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="GovAI Sentiment Lens", page_icon="📊", layout="wide")
st.title("📊 GovAI Sentiment Lens")
st.caption("POC – AI-powered public feedback analysis (persistent + frustration detection)")

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
    """Load CSV once (cached)."""
    return pd.read_csv("feedback.csv")

def save_data(df):
    """Save dataframe to CSV."""
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

# ---------- SIDEBAR CONFIG ----------
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
    new_msg = st.text_area("Citizen feedback text", help="E.g. 'Cannot login to Singpass', 'Bus timing not accurate', etc.")
    submitted = st.form_submit_button("Add to dashboard")
    if submitted and new_msg.strip():
        new_row = {
            "id": len(st.session_state.df) + 1,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "message": new_msg.strip(),
            "sentiment": local_sentiment(new_msg.strip()),
            "topic": local_topic(new_msg.strip())
        }
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
        save_data(st.session_state.df)
        st.success("✅ Feedback added and saved!")

df = st.session_state.df

# ---------- AI BATCH CLASSIFICATION ----------
st.subheader("🤖 AI Batch Classification (sentiment + topic + frustration detection)")
st.write("Classify all feedback using AI and save results back to CSV.")

if st.button("Run AI Batch Classification"):
    if client is None:
        st.warning("⚠️ No OpenAI key found, using local classification only.")
    else:
        prompt = """
        You are analyzing citizen feedback for Singapore government digital services.

        For each feedback line below, classify into:
        - Sentiment: One of [Positive, Negative, Neutral, Frustrated]
        - Topic: One of [Housing, Transport, Digital/Singpass, Social/Financial, Health, Education, Elderly/Inclusion, e-Payment, Others]

        ### Rules:
        1. **Frustrated** → Any strong irritation, repeated failures, anger, or emotional stress.
           Examples: "keeps timing out", "every time I try", "always fails", "so annoying", "this is ridiculous".
        2. **Negative** → Complaint or dissatisfaction stated factually.
        3. **Positive** → Praise, gratitude, satisfaction.
        4. **Neutral** → Objective statements or questions with no emotion.

        ### Examples:
        1. "Bus arrival time in the app is not accurate." → Negative, Transport  
        2. "Singpass keeps timing out when I try to access services." → Frustrated, Digital/Singpass  
        3. "Thanks for fixing the login issue." → Positive, Digital/Singpass  
        4. "Can I check my CPF balance?" → Neutral, Social/Financial  
        5. "I'm really angry the system keeps crashing after update." → Frustrated, Digital/Singpass  
        6. "The HDB rental form is confusing." → Negative, Housing  
        7. "The doctor appointment booking system is confusing." → Negative, Health  
        8. "The new school portal is helpful for parents." → Positive, Education

        Now classify the following feedback lines and respond ONLY in CSV format:
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

            # ---- PARSE SAFE CSV ----
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
                st.success(f"✅ Updated {len(parsed_df)} rows from AI output and saved to CSV.")
            else:
                st.warning("⚠️ No valid lines detected — AI output ignored.")
        except Exception as e:
            st.error(f"⚠️ OpenAI error: {e}")

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
st.dataframe(df[["date", "message", "sentiment", "topic"]], use_container_width=True)

# ---------- FILTER ----------
st.subheader("🔎 Filter by topic")
topic_list = ["All"] + sorted(df["topic"].unique().tolist())
selected_topic = st.selectbox("Select topic", topic_list)
if selected_topic != "All":
    st.write(df[df["topic"] == selected_topic][["date", "message", "sentiment", "topic"]])
else:
    st.write(df[["date", "message", "sentiment", "topic"]])

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
