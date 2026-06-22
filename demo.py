import asyncio
import sys
from dotenv import load_dotenv

load_dotenv()

from agents.cv_analyzer import analyze_all_cvs
from agents.interview_agent import start_interview, submit_answer
from agents.scoring_agent import rank_all_candidates
from agents.report_writer import generate_hiring_report, format_report


SAMPLE_JOB_DESCRIPTION = """
Senior AI Engineer
Requirements: LangChain, LangGraph, FastAPI, Production AI Systems, 4+ years experience
Responsibilities: Build and deploy AI agents, design multi-agent systems
"""


async def run_demo(cv_paths: list[str]) -> str:
    print(f"Analyzing {len(cv_paths)} CV(s)...")
    results = await analyze_all_cvs(cv_paths)
    successful = [r["data"] for r in results if r.get("status") == "success"]

    if not successful:
        print("No CVs could be analyzed. Stopping.")
        return ""

    interview_results = []

    for i, cv_analysis in enumerate(successful):
        print(f"\n{'=' * 50}")
        print(f"Interviewing: {cv_analysis.candidate_name}")
        print(f"{'=' * 50}")

        session_id = f"demo-candidate-{i}"
        result = await start_interview(session_id, cv_analysis, SAMPLE_JOB_DESCRIPTION)

        while not result["complete"]:
            print(f"\nQuestion: {result['question']}")
            answer = input("Your answer: ").strip() or "No answer provided"
            result = await submit_answer(session_id, answer)

        interview_results.append({
            "cv_analysis": {
                "candidate_name": cv_analysis.candidate_name,
                "years_of_experience": cv_analysis.years_of_experience,
                "technical_skills": cv_analysis.technical_skills,
                "education": cv_analysis.education,
                "previous_roles": cv_analysis.previous_roles,
                "strength_summary": cv_analysis.strength_summary,
                "weakness_summary": cv_analysis.weakness_summary
            },
            "job_description": SAMPLE_JOB_DESCRIPTION,
            "interview_qa": result["qa_pairs"]
        })

    print("\nScoring candidates...")
    ranked_scores = await rank_all_candidates(interview_results)

    print("Generating report...")
    report = await generate_hiring_report(SAMPLE_JOB_DESCRIPTION, ranked_scores)

    return format_report(report)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python demo.py path/to/cv1.pdf [path/to/cv2.pdf ...]")
        sys.exit(1)

    cv_paths = sys.argv[1:]
    report = await run_demo(cv_paths)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())