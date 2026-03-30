import streamlit as st
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json

st.set_page_config(page_title="AI Demand Matching", layout="wide")
st.title("🚀 AI Demand vs Bench Matching Tool")

# ============================
# LOAD API + MODEL
# ============================

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ============================
# OPENAI SKILL EXTRACTION (CACHED)
# ============================

@st.cache_data(ttl=86400, show_spinner=False)
def extract_skills_ai(text):

    if not text or str(text).strip() == "":
        return {"primary_skill": "", "secondary_skills": []}

    prompt = f"""
    Extract technical skills from this text.

    Identify:
    - primary_skill (main technology like React, .NET, Java, Python)
    - secondary_skills (other tools/skills)

    Text:
    {text}

    Return only JSON:
    {{
      "primary_skill": "",
      "secondary_skills": []
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        # Fix for bad JSON responses
        if content.startswith("```"):
            content = content.split("```")[1]

        return json.loads(content)

    except:
        return {"primary_skill": "", "secondary_skills": []}


# ============================
# FILE UPLOAD
# ============================

demand_file = st.file_uploader("📄 Upload Demand File", type=["xlsx"])
bench_file = st.file_uploader("📄 Upload Bench File", type=["xlsx"])

if demand_file and bench_file:

    demand_df = pd.read_excel(demand_file)
    bench_df = pd.read_excel(bench_file, sheet_name="Base Data")

    demand_df.columns = demand_df.columns.str.strip().str.lower()
    bench_df.columns = bench_df.columns.str.strip().str.lower()

    st.success("✅ Files uploaded")

    if st.button("🔍 Run Matching"):

        with st.spinner("Processing..."):

            model = load_model()

            # ============================
            # FILTER BENCH
            # ============================

            valid_status = [
                "bu active bench",
                "future release",
                "practice blocked",
                "practice active bench"
            ]

            bench_df = bench_df[
                bench_df["status"].str.lower().isin(valid_status)
            ].copy()

            # ============================
            # BUILD TEXT
            # ============================

            def build_demand_text(row):
                return " ".join([
                    str(row.get("requirement title", "")),
                    str(row.get("high level requirements", "")),
                    str(row.get("skill category", ""))
                ])

            def build_bench_text(row):
                return " ".join([
                    str(row.get("primary skills", "")),
                    str(row.get("sec. skills", "")),
                    str(row.get("practice skill group", ""))
                ])

            # ============================
            # EXTRACT SKILLS (ONCE)
            # ============================

            st.info("🔍 Extracting skills (first run may take time)...")

            demand_skill_map = {}
            for _, row in demand_df.iterrows():
                did = row.get("requisition id")
                text = build_demand_text(row)
                demand_skill_map[did] = extract_skills_ai(text)

            bench_skill_map = {}
            for _, row in bench_df.iterrows():
                eid = row.get("emp_id")
                text = build_bench_text(row)
                bench_skill_map[eid] = extract_skills_ai(text)

            # ============================
            # EMBEDDINGS
            # ============================

            demand_embeddings = {}
            for _, row in demand_df.iterrows():
                did = row.get("requisition id")
                text = build_demand_text(row)

                if did:
                    demand_embeddings[did] = model.encode(text)

            bench_embeddings = {}
            for _, row in bench_df.iterrows():
                eid = row.get("emp_id")
                text = build_bench_text(row)

                if eid:
                    bench_embeddings[eid] = model.encode(text)

            # ============================
            # MATCHING
            # ============================

            results = []

            for _, demand in demand_df.iterrows():

                did = demand.get("requisition id")
                d_emb = demand_embeddings.get(did)
                d_skills = demand_skill_map.get(did)

                # ✅ FIXED ERROR HERE
                if d_emb is None or d_skills is None:
                    continue

                for _, bench in bench_df.iterrows():

                    eid = bench.get("emp_id")
                    b_emb = bench_embeddings.get(eid)
                    b_skills = bench_skill_map.get(eid)

                    # ✅ FIXED ERROR HERE
                    if b_emb is None or b_skills is None:
                        continue

                    # ============================
                    # PRIMARY SKILL MATCH (STRICT)
                    # ============================

                    if d_skills["primary_skill"] != b_skills["primary_skill"]:
                        continue

                    # ============================
                    # AI SIMILARITY
                    # ============================

                    score = cosine_similarity([b_emb], [d_emb])[0][0]

                    if score < 0.25:
                        continue

                    results.append({
                        "Requisition ID": did,
                        "Emp_ID": eid,
                        "Name": bench.get("name"),
                        "Primary Skill": b_skills["primary_skill"],
                        "Match %": round(score * 100, 2)
                    })

            # ============================
            # OUTPUT
            # ============================

            result_df = pd.DataFrame(results)

            if result_df.empty:
                st.error("❌ No matches found")
            else:
                result_df.sort_values(
                    by=["Requisition ID", "Match %"],
                    ascending=[True, False],
                    inplace=True
                )

                top5_df = result_df.groupby("Requisition ID").head(5)

                st.success(f"✅ {len(top5_df)} matches found")
                st.dataframe(top5_df, use_container_width=True)

                file_name = f"Available_Resources_{datetime.now().strftime('%H%M%S')}.xlsx"
                top5_df.to_excel(file_name, index=False)

                with open(file_name, "rb") as f:
                    st.download_button("📥 Download Results", f, file_name)