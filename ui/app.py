import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI Hiring System",
    page_icon="AI",
    layout="wide"
)

st.title("AI Hiring System")
st.caption("Automated CV Analysis, AI Interviews & Candidate Ranking")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "current_candidate" not in st.session_state:
    st.session_state.current_candidate = None

if st.session_state.stage == "upload":
    st.header("Upload CVs")

    job_description = st.text_area("Job Description", height=100)
    uploaded_files = st.file_uploader("Upload CVs (PDF)", type="pdf", accept_multiple_files=True)

    if st.button("Start Analysis", type="primary"):
        if not job_description or not uploaded_files:
            st.error("Please add job description and at least one CV")
        else:
            with st.spinner("Uploading and analyzing CVs..."):
                files = [("files", (f.name, f.read(), "application/pdf")) for f in uploaded_files]
                upload_response = requests.post(
                    f"{API_URL}/upload-cvs",
                    data={"job_description": job_description},
                    files=files
                )

                if upload_response.status_code != 200:
                    st.error(f"Upload failed: {upload_response.text}")
                    st.stop()

                session_id = upload_response.json()["session_id"]
                st.session_state.session_id = session_id

                analyze_response = requests.post(f"{API_URL}/analyze/{session_id}")

                if analyze_response.status_code != 200:
                    st.error(f"Analysis failed: {analyze_response.text}")
                    st.stop()

                data = analyze_response.json()
                st.session_state.current_question = data["question"]
                st.session_state.current_candidate = data["current_candidate"]
                st.session_state.candidate_index = data["candidate_index"]
                st.session_state.total_candidates = data["total_candidates"]
                st.session_state.stage = "interview"
                st.rerun()

elif st.session_state.stage == "interview":
    st.header("Interview in Progress")
    st.info(
        f"Candidate {st.session_state.candidate_index + 1} of "
        f"{st.session_state.total_candidates}: **{st.session_state.current_candidate}**"
    )

    st.subheader("Question")
    st.write(st.session_state.current_question)

    answer = st.text_area("Your answer", key="answer_input", height=120)

    if st.button("Submit Answer", type="primary"):
        if not answer.strip():
            st.warning("Please write an answer before submitting")
        else:
            with st.spinner("Processing answer..."):
                response = requests.post(
                    f"{API_URL}/interview/{st.session_state.session_id}/answer",
                    data={"answer": answer}
                )

                if response.status_code != 200:
                    st.error(f"Failed to submit answer: {response.text}")
                    st.stop()

                data = response.json()

                if data["status"] == "interviewing":
                    st.session_state.current_question = data["question"]
                    st.session_state.current_candidate = data["current_candidate"]
                    st.session_state.candidate_index = data["candidate_index"]
                    st.rerun()

                elif data["status"] == "scoring":
                    st.session_state.stage = "scoring"
                    st.rerun()

elif st.session_state.stage == "scoring":
    st.header("Generating Report")
    st.info("All interviews complete. Scoring candidates and writing the report...")

    if st.button("Check if report is ready"):
        status = requests.get(f"{API_URL}/status/{st.session_state.session_id}").json()

        if status["status"] == "complete":
            st.session_state.stage = "complete"
            st.rerun()
        elif status["status"] == "failed":
            st.error(f"Pipeline failed: {status.get('error')}")
        else:
            st.warning("Still processing, click again in a few seconds")

elif st.session_state.stage == "complete":
    st.header("Final Report")

    report = requests.get(f"{API_URL}/report/{st.session_state.session_id}").json()
    st.text(report["report"])

    if st.button("Start a New Hiring Round"):
        st.session_state.session_id = None
        st.session_state.stage = "upload"
        st.session_state.current_question = None
        st.rerun()