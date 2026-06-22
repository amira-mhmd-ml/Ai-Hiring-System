import asyncio
import fitz  
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class CVAnalysis(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    years_of_experience: int = Field(description="Total years of professional experience")
    technical_skills: List[str] = Field(description="List of all technical skills mentioned")
    education: str = Field(description="Highest education degree and field of study")
    previous_roles: List[str] = Field(description="List of previous job titles held")
    strength_summary: str = Field(description="2-3 sentences summarizing candidate strengths")
    weakness_summary: str = Field(description="2-3 sentences summarizing candidate weaknesses or gaps")


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return full_text


def build_cv_analyzer_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert HR analyst with 10+ years of experience.
            Your job is to extract structured information from CVs accurately.
            
            Rules:
            - Be objective and precise
            - Do NOT assume information not explicitly mentioned
            - For years of experience, calculate from dates if possible
            - List ALL technical skills mentioned, including tools and frameworks
            - Be honest about weaknesses based on what's missing vs job market expectations
            """
        ),
        (
            "human",
            "Please analyze this CV and extract structured information:\n\n{cv_text}"
        )
    ])

    return prompt | llm.with_structured_output(CVAnalysis)


async def call_with_retry(chain, inputs: dict, max_retries: int = 3) -> CVAnalysis:
    base_delay = 1

    for attempt in range(max_retries):
        try:
            return await chain.ainvoke(inputs)

        except Exception as e:
            is_last_attempt = attempt == max_retries - 1

            if is_last_attempt:
                raise e

            wait_time = base_delay * (2 ** attempt)
            print(f"  [Retry] Attempt {attempt + 1} failed. Retrying in {wait_time}s... Error: {e}")
            await asyncio.sleep(wait_time)


async def analyze_single_cv(
    pdf_path: str,
    semaphore: asyncio.Semaphore,
    rate_limiter: asyncio.Semaphore
) -> dict:
    async with semaphore:
        try:
            cv_text = extract_text_from_pdf(pdf_path)

            if len(cv_text.strip()) < 100:
                return {
                    "path": pdf_path,
                    "status": "failed",
                    "reason": "PDF appears to be scanned image or empty — needs OCR"
                }

            async with rate_limiter:
                chain = build_cv_analyzer_chain()
                result = await call_with_retry(chain, {"cv_text": cv_text})

                await asyncio.sleep(0.5)

                return {
                    "path": pdf_path,
                    "status": "success",
                    "data": result
                }

        except ValueError as e:
            return {"path": pdf_path, "status": "failed", "reason": f"PDF Error: {e}"}

        except Exception as e:
            return {"path": pdf_path, "status": "failed", "reason": f"API Error after retries: {e}"}


async def analyze_all_cvs(pdf_paths: List[str]) -> List[dict]:
    # قمنا بتقليل الـ Concurrency والـ Rate limit قليلاً ليتناسب مع الـ Free Tier لـ Gemini وتجنب الـ Quota limits
    MAX_CONCURRENT = 3
    RATE_LIMIT = 3

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    rate_limiter = asyncio.Semaphore(RATE_LIMIT)

    tasks = [
        analyze_single_cv(path, semaphore, rate_limiter)
        for path in pdf_paths
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    failed = [r for r in results if isinstance(r, dict) and r.get("status") == "failed"]

    print(f"\n[CV Analyzer]  Success: {len(successful)} | ❌ Failed: {len(failed)}")

    for f in failed:
        print(f"  └─ {f['path']}: {f['reason']}")

    return list(results)


async def analyze_cv(pdf_path: str) -> CVAnalysis | None:
    semaphore = asyncio.Semaphore(1)
    rate_limiter = asyncio.Semaphore(1)
    result = await analyze_single_cv(pdf_path, semaphore, rate_limiter)

    if result["status"] == "success":
        return result["data"]

    print(f"[CV Analyzer] Failed: {result['reason']}")
    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cv_analyzer.py path/to/cv.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Analyzing: {pdf_path}")

    result = asyncio.run(analyze_cv(pdf_path))

    if result:
        print(f"\n{'='*50}")
        print(f"Name       : {result.candidate_name}")
        print(f"Experience : {result.years_of_experience} years")
        print(f"Education  : {result.education}")
        print(f"Skills     : {', '.join(result.technical_skills)}")
        print(f"Roles      : {', '.join(result.previous_roles)}")
        print(f"Strengths  : {result.strength_summary}")
        print(f"Weaknesses : {result.weakness_summary}")