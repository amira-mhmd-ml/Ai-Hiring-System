
import uuid
import shutil
import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from dotenv import load_dotenv

from agents.cv_analyzer import analyze_all_cvs
from agents.interview_agent import start_interview, submit_answer
from agents.scoring_agent import rank_all_candidates
from agents.report_writer import generate_hiring_report, format_report

load_dotenv()


app = FastAPI(
    title="AI Hiring System",
    description="Automated CV analysis, AI interviews, and candidate ranking",
    version="1.1.0"
)


sessions: dict = {}


@app.post("/upload-cvs")
async def upload_cvs(
    job_description: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    session_id = str(uuid.uuid4())
    session_dir = f"uploads/{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    saved_paths = []
    rejected_files = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            rejected_files.append(file.filename)
            continue

        file_path = f"{session_dir}/{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(file_path)

    if not saved_paths:
        raise HTTPException(
            status_code=400,
            detail=f"No valid PDF files found. Rejected: {rejected_files}"
        )

    sessions[session_id] = {
        "job_description": job_description,
        "cv_paths": saved_paths,
        "status": "uploaded",
        "cv_analyses": [],         
        "current_candidate_index": 0,
        "interview_results": [],    
        "report": None,
        "error": None
    }

    return {
        "session_id": session_id,
        "uploaded": len(saved_paths),
        "rejected": rejected_files,
        "message": f"Successfully uploaded {len(saved_paths)} CV(s)",
        "next_step": f"POST /analyze/{session_id}"
    }


@app.post("/analyze/{session_id}")
async def start_analysis(session_id: str):

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    if session["status"] != "uploaded":
        return {
            "message": f"Session already in status: {session['status']}",
            "status": session["status"]
        }

    session["status"] = "analyzing_cvs"

    results = await analyze_all_cvs(session["cv_paths"])
    successful = [r["data"] for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]

    session["cv_analyses"] = successful

    if not successful:
        session["status"] = "failed"
        session["error"] = f"No CVs could be analyzed. Failures: {failed}"
        raise HTTPException(status_code=400, detail=session["error"])

  
    session["status"] = "interviewing"
    first_candidate = session["cv_analyses"][0]
    interview_session_id = f"{session_id}-candidate-0"

    result = await start_interview(
        session_id=interview_session_id,
        cv_analysis=first_candidate,
        job_description=session["job_description"]
    )

    return {
        "session_id": session_id,
        "status": "interviewing",
        "cvs_analyzed": len(successful),
        "cvs_failed": len(failed),
        "current_candidate": first_candidate.candidate_name,
        "candidate_index": 0,
        "total_candidates": len(successful),
        "question": result["question"],
        "next_step": f"POST /interview/{session_id}/answer"
    }



@app.post("/interview/{session_id}/answer")
async def answer_interview_question(
    session_id: str,
    background_tasks: BackgroundTasks,
    answer: str = Form(...)
):

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    if session["status"] != "interviewing":
        raise HTTPException(
            status_code=400,
            detail=f"Session is not in interview stage. Current status: {session['status']}"
        )

    candidate_index = session["current_candidate_index"]
    interview_session_id = f"{session_id}-candidate-{candidate_index}"

    result = await submit_answer(interview_session_id, answer)

    if not result["complete"]:
        
        return {
            "status": "interviewing",
            "current_candidate": session["cv_analyses"][candidate_index].candidate_name,
            "candidate_index": candidate_index,
            "total_candidates": len(session["cv_analyses"]),
            "question": result["question"]
        }

    
    current_cv = session["cv_analyses"][candidate_index]
    session["interview_results"].append({
        "cv_analysis": {
            "candidate_name": current_cv.candidate_name,
            "years_of_experience": current_cv.years_of_experience,
            "technical_skills": current_cv.technical_skills,
            "education": current_cv.education,
            "previous_roles": current_cv.previous_roles,
            "strength_summary": current_cv.strength_summary,
            "weakness_summary": current_cv.weakness_summary
        },
        "job_description": session["job_description"],
        "interview_qa": result["qa_pairs"]
    })

    next_index = candidate_index + 1

    if next_index < len(session["cv_analyses"]):
       
        session["current_candidate_index"] = next_index
        next_candidate = session["cv_analyses"][next_index]
        next_interview_session_id = f"{session_id}-candidate-{next_index}"

        next_result = await start_interview(
            session_id=next_interview_session_id,
            cv_analysis=next_candidate,
            job_description=session["job_description"]
        )

        return {
            "status": "interviewing",
            "current_candidate": next_candidate.candidate_name,
            "candidate_index": next_index,
            "total_candidates": len(session["cv_analyses"]),
            "question": next_result["question"],
            "message": f"Finished interviewing {current_cv.candidate_name}. Starting next candidate."
        }

    
    session["status"] = "scoring"

    async def finish_pipeline():
        try:
            ranked_scores = await rank_all_candidates(session["interview_results"])
            report = await generate_hiring_report(
                job_description=session["job_description"],
                ranked_candidates=ranked_scores
            )
            session["report"] = format_report(report)
            session["status"] = "complete"
        except Exception as e:
            session["status"] = "failed"
            session["error"] = str(e)

    background_tasks.add_task(finish_pipeline)

    return {
        "status": "scoring",
        "message": "All interviews complete. Generating report...",
        "next_step": f"GET /status/{session_id}"
    }



@app.get("/status/{session_id}")
async def get_status(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    status = session["status"]

    response = {"session_id": session_id, "status": status}

    if status == "complete":
        response["next_step"] = f"GET /report/{session_id}"
    elif status == "failed":
        response["error"] = session.get("error", "Unknown error")
    elif status == "scoring":
        response["message"] = "Scoring candidates and generating report, check again shortly"
    elif status == "interviewing":
        idx = session["current_candidate_index"]
        response["message"] = f"Interview in progress for candidate {idx + 1}/{len(session['cv_analyses'])}"

    return response



@app.get("/report/{session_id}")
async def get_report(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    if session["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Report not ready. Current status: {session['status']}"
        )

    return {
        "session_id": session_id,
        "status": "complete",
        "report": session["report"]
    }



@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_sessions": len(sessions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)