import asyncio
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
import os
class CandidateScore(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    technical_score: float = Field(description="Technical knowledge and skills score 0-10")
    experience_score: float = Field(description="Relevance of experience to the role 0-10")
    communication_score: float = Field(description="Clarity and depth of communication 0-10")
    job_fit_score: float = Field(description="Overall CV match with job requirements 0-10")
    overall_score: float = Field(description="Final weighted score 0-10")
    strengths: List[str] = Field(description="Top 3 specific strengths for this role")
    gaps: List[str] = Field(description="Top 3 gaps compared to job requirements")
    recommendation: str = Field(description="One of: Strong Yes / Yes / Maybe / No")
    reasoning: str = Field(description="2-3 sentences explaining the overall score")


async def score_candidate(
    cv_analysis: dict,
    job_description: str,
    interview_qa: List[dict]
) -> CandidateScore:

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
)




    formatted_qa = "\n\n".join([
        f"Q{i+1}: {item['question']}\nA{i+1}: {item['answer']}"
        for i, item in enumerate(interview_qa)
    ])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an objective senior hiring evaluator.
            
            Score using these EXACT weights:
            - Interview Performance : 50%  (did they PROVE real knowledge?)
            - CV match with Job     : 30%  (how relevant is their background?)
            - Experience relevance  : 20%  (quality of experience, not just years)
            
            STRICT RULES:
            - Be objective, not generous — a 10 means exceptional, 5 means average
            - A great CV with weak interview answers = low score (interview is 50%)
            - Compare answers to job requirements, not general knowledge
            - Document specific gaps, not vague observations
            - Recommendation scale:
                Strong Yes → 8.5+
                Yes        → 7.0-8.4
                Maybe      → 5.0-6.9
                No         → below 5.0
            """
        ),
        (
            "human",
            """
            ── JOB REQUIREMENTS ─────────────────────────────
            {job_description}
            
            ── CANDIDATE BACKGROUND ─────────────────────────
            {cv_analysis}
            
            ── INTERVIEW PERFORMANCE ────────────────────────
            {formatted_qa}
            
            Provide a fair, weighted evaluation:
            """
        )
    ])

    chain = prompt | llm.with_structured_output(CandidateScore)

    return await chain.ainvoke({
        "job_description": job_description,
        "cv_analysis": str(cv_analysis),
        "formatted_qa": formatted_qa
    })


async def rank_all_candidates(candidates: List[dict]) -> List[CandidateScore]:
    tasks = [
        score_candidate(
            cv_analysis=c["cv_analysis"],
            job_description=c["job_description"],
            interview_qa=c["interview_qa"]
        )
        for c in candidates
    ]

    scores = await asyncio.gather(*tasks, return_exceptions=True)
    valid_scores = [s for s in scores if isinstance(s, CandidateScore)]

    return sorted(valid_scores, key=lambda x: x.overall_score, reverse=True)


def print_scores(ranked_candidates: List[CandidateScore]) -> None:
    print(f"\n{'='*55}")
    print("CANDIDATE RANKING")
    print(f"{'='*55}")

    for rank, c in enumerate(ranked_candidates, 1):
        bar = "█" * int(c.overall_score) + "░" * (10 - int(c.overall_score))
        print(f"""
#{rank} {c.candidate_name}
   Score        : {bar} {c.overall_score:.1f}/10
   Technical    : {c.technical_score:.1f} | Job Fit: {c.job_fit_score:.1f} | Comm: {c.communication_score:.1f}
   Recommend    : {c.recommendation}
   Strengths    : {' | '.join(c.strengths)}
   Gaps         : {' | '.join(c.gaps)}
   Reasoning    : {c.reasoning}
        """)


if __name__ == "__main__":
    sample_candidates = [
        {
            "cv_analysis": {
                "candidate_name": "Sara Ahmed",
                "years_of_experience": 3,
                "technical_skills": ["Python", "LangChain", "FastAPI", "RAG"],
                "strength_summary": "Strong LLM application development",
                "weakness_summary": "Limited production scale experience"
            },
            "job_description": "Senior AI Engineer — LangGraph, LLMs, FastAPI, Production Systems",
            "interview_qa": [
                {"question": "Explain RAG architecture", "answer": "RAG retrieves relevant documents from a vector database then passes them as context to the LLM to generate grounded answers. It solves hallucination problems."},
                {"question": "How do you handle API rate limits?", "answer": "I use exponential backoff — retry after 1s, 2s, 4s. Also async processing with semaphores to control concurrent calls."}
            ]
        },
        {
            "cv_analysis": {
                "candidate_name": "Ahmed Hassan",
                "years_of_experience": 5,
                "technical_skills": ["Python", "TensorFlow", "Docker"],
                "strength_summary": "Strong traditional ML background",
                "weakness_summary": "No LLM or RAG experience"
            },
            "job_description": "Senior AI Engineer — LangGraph, LLMs, FastAPI, Production Systems",
            "interview_qa": [
                {"question": "Explain RAG architecture", "answer": "I haven't worked with RAG before, but I think it's related to retrieval somehow."},
                {"question": "How do you handle API rate limits?", "answer": "Maybe add some delays? I'm not sure about the best approach."}
            ]
        }
    ]

    ranked = asyncio.run(rank_all_candidates(sample_candidates))
    print_scores(ranked)