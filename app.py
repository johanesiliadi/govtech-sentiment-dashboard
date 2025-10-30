import streamlit as st
import pandas as pd
from datetime import datetime
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="GovAI Sentiment Lens", page_icon="📊", layout="wide")
st.title("📊 GovAI Sentiment Lens")
st.caption("POC – AI-powered public feedback analysis (CSV + manual input + batch AI classification)")

# ---------- OPENAI CLIENT (optional) ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
else:
    client = None

# ---------- 1. LOAD BASE DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv("feedback.csv")

df = load_data()

# ---------- 2. MANUAL INPUT SECTION ----------
st.subheader("➕ Add new citizen feedback")
with st.form("add_form", clear_on_submit=True):
    new_msg = st.text_area("Citizen feedback text", help="E.g. 'Cannot login to Singpass', 'Thanks for fast response', etc.")
    submitted = st.form_submit_button("Add to dashboard")
    if submitted and new_msg.strip():
        new_row = {
            "id": len(df) + 1,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "message": new_msg.strip()
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.success("✅ Feedback added below.")

# ---------- 3. LOCAL FALLBACK CLASSIFIERS ----------
def local_sentiment(text: str) -> str:
    t = text.lower()
    neg = ["cannot", "not working", "slow", "error", "complain", "bad", "fail", "issue", "problem", "frustrating", "timeout", "crash"]
    pos = ["thank", "thanks", "good", "great", "appreciate", "helpful", "well done", "improvement", "love"]
    if any(w in t for w in neg):
        return "Negative"
    if any(w in t for w in pos):
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
    return "Others"

df["sentiment"] = df["message"].apply(local_sentiment)
df["topic"] = df["message"].apply(local_topic)

# ---------- 4. BATCH AI CLASSIFICATION ----------
st.subheader("🤖 AI Batch Classification (sentiment + topic)")
st.write("Classify all feedback at once using a single AI call. If no API key, fallback to local rules.")

if st.button("Run AI Batch Classification"):
    if client is None:
        st.warning("⚠️ No OpenAI key found, using local classification only.")
    else:
        # Build batch prompt
        prompt = "Classify each feedback line below as Sentiment (Positive, Negative, Neutral) and Topic.\n"
        prompt += "Topics: [Housing, Transport, Digital/Singpass, Social/Financial, Health, Elderly/Inclusion, e-Payment, Others, Frustated People]\n\n"
        for i, msg in enumerate(df["message"], start=1):
            prompt += f"{i}. {msg}\n"
        prompt += "\nRespond ONLY in CSV format: id,sentiment,topic"

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            output = resp.choices[0].message.content.strip()
            st.text_area("AI Batch Output (CSV)", output, height=150)

            # Parse AI output into dataframe
            lines = [l.strip() for l in output.splitlines() if "," in l]
            sentiments, topics = [], []
            for idx, line in enumerate(lines):
                parts = line.split(",")
                if len(parts) >= 3:
                    sentiments.append(parts[1].strip().title())
                    topics.append(parts[2].strip())
            if len(sentiments) == len(df):
                df["sentiment"] = sentiments
                df["topic"] = topics
                st.success("✅ AI classification updated.")
            else:
                st.warning("⚠️ Could not map all lines — showing raw output above.")
        except Exception as e:
            st.error(f"⚠️ OpenAI error: {e}")

# ---------- 5. DASHBOARD AREA ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Sentiment distribution")
    st.bar_chart(df["sentiment"].value_counts())
with col2:
    st.subheader("🏷️ Topic distribution")
    st.bar_chart(df["topic"].value_counts())

st.subheader("📋 Feedback table")
st.dataframe(df[["date", "message", "sentiment", "topic"]], use_container_width=True)

# ---------- 6. FILTER ----------
st.subheader("🔎 Filter by topic")
topic_list = ["All"] + sorted(df["topic"].unique().tolist())
selected_topic = st.selectbox("Select topic", topic_list)
if selected_topic != "All":
    st.write(df[df["topic"] == selected_topic][["date", "message", "sentiment", "topic"]])
else:
    st.write(df[["date", "message", "sentiment", "topic"]])

# ---------- 7. AI EXECUTIVE SUMMARY ----------
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
        3) Which agency or team is likely impacted
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
