import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

st.set_page_config(page_title="Demand vs Bench Matching", layout="wide")

st.title("🚀 GenAI Demand vs Bench Matching Tool")

# Upload files
demand_file = st.file_uploader("📄 Upload Demand File", type=["xlsx"])
bench_file = st.file_uploader("📄 Upload Bench File", type=["xlsx"])

if demand_file and bench_file:

    demand_df = pd.read_excel(demand_file)
    bench_df = pd.read_excel(bench_file, sheet_name="Base Data")

    # Normalize column names
    demand_df.columns = demand_df.columns.str.strip().str.lower()
    bench_df.columns = bench_df.columns.str.strip().str.lower()

    st.success("✅ Files uploaded successfully")

    if st.button("🔍 Run Matching"):

        with st.spinner("Running AI Matching..."):

            # ============================================================
            # FILTER BENCH
            # ============================================================

            valid_status = [
                "bu active bench",
                "future release",
                "practice blocked",
                "practice active bench"
            ]

            bench_df = bench_df[
                bench_df["status"].str.lower().isin(valid_status)
            ].copy()

            # ============================================================
            # NORMALIZATION
            # ============================================================

            def normalize(text):
                if pd.isna(text):
                    return ""

                text = str(text).lower()
                text = text.replace("dot net", ".net")
                text = text.replace("asp.net", ".net")
                text = text.replace(".net core", ".net")
                text = text.replace("vb.net", ".net")
                text = text.replace("sql server", "sql")

                return text.replace(",", " ").replace("/", " ")

            # ============================================================
            # TEXT BUILDERS
            # ============================================================

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
                    normalize(row.get("current location")),
                    normalize(row.get("location preference"))
                ]))

            # ============================================================
            # SKILL MATCH
            # ============================================================

            def skill_match_score(demand_text, bench_text):
                demand_words = set(demand_text.split())
                bench_words = set(bench_text.split())

                ignore = {"and", "or", "with", "the", "a", "in"}
                demand_words -= ignore
                bench_words -= ignore

                common = demand_words.intersection(bench_words)

                return len(common) / (len(demand_words) + 1)

            # ============================================================
            # LOAD MODEL
            # ============================================================

            model = SentenceTransformer("all-MiniLM-L6-v2")

            # ============================================================
            # EMBEDDINGS
            # ============================================================

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

            # ============================================================
            # MATCHING
            # ============================================================

            SIMILARITY_THRESHOLD = 0.20
            results = []

            for _, demand in demand_df.iterrows():

                did = demand.get("requisition id")
                d_emb = demand_embeddings.get(did)

                if d_emb is None:
                    continue

                demand_text = build_demand_text(demand)
                demand_exp = demand.get("experience")
                demand_loc = demand.get("location")

                for _, bench in bench_df.iterrows():

                    eid = bench.get("emp_id")
                    b_emb = bench_embeddings.get(eid)

                    if b_emb is None:
                        continue

                    bench_text = build_bench_text(bench)

                    ai_score = cosine_similarity([b_emb], [d_emb])[0][0]
                    skill_score = skill_match_score(demand_text, bench_text)

                    if ai_score < SIMILARITY_THRESHOLD and skill_score < 0.05:
                        continue

                    # Final score
                    final_score = (ai_score * 0.5) + (skill_score * 0.5)

                    results.append({
                        "Requisition ID": did,
                        "Emp_ID": eid,
                        "Name": bench.get("name"),
                        "Primary Skills": bench.get("primary skills"),
                        "Location": bench.get("current location"),
                        "AI Score %": round(ai_score * 100, 2),
                        "Skill Match %": round(skill_score * 100, 2),
                        "Final Score %": round(final_score * 100, 2)
                    })

            # ============================================================
            # OUTPUT
            # ============================================================

            result_df = pd.DataFrame(results)

            if result_df.empty:
                st.error("❌ No matches found")
            else:
                result_df.sort_values(
                    by=["Requisition ID", "Final Score %"],
                    ascending=[True, False],
                    inplace=True
                )

                top5_df = result_df.groupby("Requisition ID").head(5)

                st.success(f"✅ {len(top5_df)} matches found")
                st.dataframe(top5_df)

                file_name = f"Available_Resources_{datetime.now().strftime('%H%M%S')}.xlsx"
                top5_df.to_excel(file_name, index=False)

                with open(file_name, "rb") as f:
                    st.download_button("📥 Download Results", f, file_name)