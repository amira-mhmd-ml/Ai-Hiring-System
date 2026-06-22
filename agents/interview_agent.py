
import asyncio
import os
from typing import List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.cv_analyzer import CVAnalysis


class InterviewState(TypedDict):
    candidate_name: str
    cv_analysis: dict
    job_description: str
    questions_asked: List[str]
    answers_given: List[str]
    current_question: str
    interview_complete: bool
    qa_pairs: List[dict]


async def generate_question_node(state: InterviewState) -> InterviewState:

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )

    questions_so_far = "\n".join(state["questions_asked"]) or "None yet"
    last_answer = state["answers_given"][-1] if state["answers_given"] else "Interview just started"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a professional senior technical interviewer.
            Generate ONE smart, specific interview question.

            Rules:
            - Ask only ONE question per turn
            - Make it specific to this candidate's background AND the job requirements
            - Don't repeat topics already covered
            - If last answer was shallow, go deeper on that topic
            - If 5 questions have been asked, return exactly: INTERVIEW_COMPLETE
            - Questions should test real understanding, not just definitions
            """
        ),
        (
            "human",
            """
            Job Requirements: {job_description}

            Candidate Background: {cv_analysis}

            Questions Asked So Far:
            {questions_so_far}

            Last Answer Given:
            {last_answer}

            Generate the next interview question:
            """
        )
    ])

    chain = prompt | llm
    response = await chain.ainvoke({
        "job_description": state["job_description"],
        "cv_analysis": str(state["cv_analysis"]),
        "questions_so_far": questions_so_far,
        "last_answer": last_answer
    })

    question = response.content.strip()

    if "INTERVIEW_COMPLETE" in question:
        qa_pairs = [
            {"question": q, "answer": a}
            for q, a in zip(state["questions_asked"], state["answers_given"])
        ]
        return {**state, "interview_complete": True, "qa_pairs": qa_pairs}

    return {
        **state,
        "current_question": question,
        "questions_asked": state["questions_asked"] + [question]
    }


def should_continue_interview(state: InterviewState) -> str:
    if state["interview_complete"]:
        return "end"
    return "end"  

def build_interview_graph():
    graph = StateGraph(InterviewState)
    graph.add_node("generate_question", generate_question_node)
    graph.set_entry_point("generate_question")
    graph.add_edge("generate_question", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


_interview_graph = None


def get_interview_graph():
    global _interview_graph
    if _interview_graph is None:
        _interview_graph = build_interview_graph()
    return _interview_graph



async def start_interview(
    session_id: str,
    cv_analysis: CVAnalysis,
    job_description: str
) -> dict:

    graph = get_interview_graph()

    initial_state: InterviewState = {
        "candidate_name": cv_analysis.candidate_name,
        "cv_analysis": {
            "name": cv_analysis.candidate_name,
            "experience": cv_analysis.years_of_experience,
            "skills": cv_analysis.technical_skills,
            "education": cv_analysis.education,
            "roles": cv_analysis.previous_roles,
            "strengths": cv_analysis.strength_summary,
            "weaknesses": cv_analysis.weakness_summary
        },
        "job_description": job_description,
        "questions_asked": [],
        "answers_given": [],
        "current_question": "",
        "interview_complete": False,
        "qa_pairs": []
    }

    config = {"configurable": {"thread_id": session_id}}
    result_state = await graph.ainvoke(initial_state, config)

    return {
        "question": result_state["current_question"],
        "complete": result_state["interview_complete"],
        "question_number": len(result_state["questions_asked"])
    }


async def submit_answer(session_id: str, answer: str) -> dict:

    graph = get_interview_graph()
    config = {"configurable": {"thread_id": session_id}}

    current_snapshot = await graph.aget_state(config)
    current_state = current_snapshot.values

    updated_state = {
        **current_state,
        "answers_given": current_state["answers_given"] + [answer]
    }

    result_state = await graph.ainvoke(updated_state, config)

    if result_state["interview_complete"]:
        return {
            "complete": True,
            "qa_pairs": result_state["qa_pairs"]
        }

    return {
        "complete": False,
        "question": result_state["current_question"],
        "question_number": len(result_state["questions_asked"])
    }


if __name__ == "__main__":
    async def terminal_demo():
        sample_cv = CVAnalysis(
            candidate_name="Sara Ahmed",
            years_of_experience=3,
            technical_skills=["Python", "LangChain", "FastAPI", "PostgreSQL"],
            education="BSc Computer Science",
            previous_roles=["AI Engineer", "Backend Developer"],
            strength_summary="Strong in LLM applications and API development",
            weakness_summary="Limited experience with large-scale distributed systems"
        )

        job_desc = """
        Senior AI Engineer
        Requirements: LangChain, LangGraph, FastAPI, Production AI Systems, 4+ years experience
        """

        session_id = "terminal-test-session"
        result = await start_interview(session_id, sample_cv, job_desc)

        while not result["complete"]:
            print(f"\nQuestion {result['question_number']}: {result['question']}")
            answer = input("Your answer: ").strip()
            result = await submit_answer(session_id, answer)

        print("\nInterview complete. Q&A pairs:")
        for pair in result["qa_pairs"]:
            print(f"Q: {pair['question']}\nA: {pair['answer']}\n")

    asyncio.run(terminal_demo())