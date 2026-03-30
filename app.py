import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

st.set_page_config(page_title="Demand vs Bench Matching", layout="wide")

st.title("🚀 GenAI Demand vs Bench Matching Tool")


# ============================
# SKILL EXTRACTION (NO HARDCODE)
# ============================

def extract_skills(text):
    if not text:
        return []

    text = str(text).lower()
    words = re.findall(r'\b[a-zA-Z\.\#\+]+\b', text)

    stop_words = {"and", "or", "with", "the", "in", "for", "to"}

    return [w for w in words if w not in stop_words and len(w) > 2]


# ============================
# LOAD MODEL (CACHED)
# ============================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ============================
# FILE UPLOAD
# ============================

demand_file = st.file_uploader("📄 Upload Demand File", type=["xlsx"])
bench_file = st.file_uploader("📄 Upload Bench File", type=["xlsx"])

if demand_file and bench_file:

    demand_df = pd.read_excel(demand_file)
    bench_df = pd.read_excel(bench_file, sheet_name="Base Data")

    # normalize columns
    demand_df.columns = demand_df.columns.str.strip().str.lower()
    bench_df.columns = bench_df.columns.str.strip().str.lower()

    st.success("✅ Files uploaded successfully")

    if st.button("🔍 Run Matching"):

        with st.spinner("Running AI Matching..."):

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
            # TEXT BUILDERS
            # ============================

            def normalize(text):
                if pd.isna(text):
                    return ""
                text = str(text).lower()
                text = text.replace("dot net", ".net")
                text = text.replace("asp.net", ".net")
                text = text.replace(".net core", ".net")
                text = text.replace("vb.net", ".net")
                text = text.replace("sql server", "sql")
                return text


            def build_demand_text(row):
                return " ".join(filter(None, [
                    normalize(row.get("requirement title")),
                    normalize(row.get("skill category")),
                    normalize(row.get("high level requirements")),
                    normalize(row.get("location"))
                ]))


            def build_bench_text(row):
                return " ".join(filter(None, [
                    normalize(row.get("primary skills")),
                    normalize(row.get("sec. skills")),
                    normalize(row.get("practice skill group")),
                    normalize(row.get("exp")),
                    normalize(row.get("current location"))
                ]))


            # ============================
            # CREATE EMBEDDINGS
            # ============================

            demand_embeddings = {}
            for _, row in demand_df.iterrows():
                did = row.get("requisition id")
                text = build_demand_text(row)

                if did and len(text.strip()) > 10:
                    demand_embeddings[did] = model.encode(text)

            bench_embeddings = {}
            for _, row in bench_df.iterrows():
                eid = row.get("emp_id")
                text = build_bench_text(row)

                if eid and len(text.strip()) > 10:
                    bench_embeddings[eid] = model.encode(text)

            # ============================
            # MATCHING LOGIC (IMPROVED)
            # ============================

            results = []
            SIMILARITY_THRESHOLD = 0.20

            for _, demand in demand_df.iterrows():

                did = demand.get("requisition id")
                d_emb = demand_embeddings.get(did)

                if d_emb is None:
                    continue

                demand_text = build_demand_text(demand)
                demand_skills = extract_skills(demand_text)

                for _, bench in bench_df.iterrows():

                    eid = bench.get("emp_id")
                    b_emb = bench_embeddings.get(eid)

                    if b_emb is None:
                        continue

                    bench_text = build_bench_text(bench)
                    bench_skills = extract_skills(bench_text)

                    # ============================
                    # 🔥 SKILL VALIDATION (KEY FIX)
                    # ============================

                    common_skills = set(demand_skills).intersection(set(bench_skills))

                    # 🔴 FILTER BAD MATCHES
                    if len(common_skills) < 2:
                        continue

                    # ============================
                    # AI SIMILARITY
                    # ============================

                    ai_score = cosine_similarity([b_emb], [d_emb])[0][0]

                    if ai_score < SIMILARITY_THRESHOLD:
                        continue

                    final_score = ai_score

                    results.append({
                        "Requisition ID": did,
                        "Emp_ID": eid,
                        "Name": bench.get("name"),
                        "Primary Skills": bench.get("primary skills"),
                        "Location": bench.get("current location"),
                        "Common Skills": ", ".join(common_skills),
                        "Match %": round(final_score * 100, 2)
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