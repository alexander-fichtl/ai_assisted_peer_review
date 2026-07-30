###--- Imports ----------------------------------------------------------------------------------------###
import re
from openai import OpenAI
from transformers import (
    AutoTokenizer
)
from vllm import LLM
###----------------------------------------------------------------------------------------------------###


# Have to do this:
#python3 -m vllm.entrypoints.openai.api_server \
#  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#  --dtype auto \
#  --max-model-len 32768
# Because not enough storage for this:
#from huggingface_hub import snapshot_download
#snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
#--model /path/to/downloaded/model


#python3 -m vllm.entrypoints.openai.api_server \
#  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#  --max-model-len 32768 \
#  --dtype auto



###--- Variables --------------------------------------------------------------------------------------###
VAR_filename = "testpdf"
PATH_pdf = VAR_filename + ".pdf"
PATH_output_review = VAR_filename + "_review.txt"
PATH_output_prompt = VAR_filename + "_prompt.txt"

review_questions = [
    ("Q1 (Summary):", "Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary."),
    ("Q2 (Soundness):", "Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence: 4 excellent, 3 good, 2 fair, 1 poor."),
    ("Q3 (Presentation):", "Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work: 4 excellent, 3 good, 2 fair, 1 poor."),
    ("Q4 (Contribution):", "Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader NeurIPS community? 4 excellent, 3 good, 2 fair, 1 poor."),
    ("Q5 (Strengths):", "Please give a substantive assessment of the strengths of the paper, touching on each of the following dimensions: originality, quality, clarity, and significance. We encourage reviewers to be broad in their definitions of originality and significance. For example, originality may arise from a new definition or problem formulation, creative combinations of existing ideas, application to a new domain, or removing limitations from prior results. You can incorporate Markdown and Latex into your review."),
    ("Q6 (Weaknesses):", "Please give a substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc."),
    ("Q7 (Suggestions)", "Please list and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This is important for a productive rebuttal and discussion phase with the authors."),
    ("Q8 (Ethics):", "If there are ethical issues with this paper, please flag the paper for an ethics review and select area of expertise that would be most useful for the ethics reviewer to have: No ethics review needed, Ethics review needed: Discrimination / Bias / Fairness Concerns, Ethics review needed: Inadequate Data and Algorithm Evaluation, Ethics review needed: Inappropriate Potential Applications & Impact  (e.g., human rights concerns), Ethics review needed: Privacy and Security (e.g., consent, surveillance, data storage concern), Ethics review needed: Compliance (e.g., GDPR, copyright, license, terms of use), Ethics review needed: Research Integrity Issues (e.g., plagiarism), Ethics review needed: Responsible Research Practice (e.g., IRB, documentation, research ethics), Ethics review needed: Failure to comply with NeurIPS Code of Ethics (lack of required documentation, safeguards, disclosure, licenses, legal compliance)"),
    ("Q9 (Overall):", "APlease provide an 'overall score' for this submission: 10: Award quality: Technically flawless paper with groundbreaking impact, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations. 9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI/ML and excellent impact on multiple areas of AI/ML, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations. 8: Strong Accept: Technically strong paper, with novel ideas, excellent impact on at least one area, or high-to-excellent impact on multiple areas, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations. 7: Accept: Technically solid paper, with high impact on at least one sub-area, or moderate-to-high impact on more than one areas, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations. 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations. 5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly. 4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly. 3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations. 2: Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations. 1: Very Strong Reject: For instance, a paper with trivial results or unaddressed ethical considerations."),
    ("Q10 (Confidence):", "Please provide a 'confidence score' for your assessment of this submission to indicate how confident you are in your evaluation. 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully. 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked. 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked. 1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked."),
    ("Q11 (Conduct):", "If there are no violations of the Code of Conduct with this paper, please respond with NO. Otherwise, if this paper violates the Code of Conduct, please indicate the relevant section(s) from the following options: 'Yes, Harassment, bullying, or discrimination based on personal characteristics', 'Yes, Inappropriate physical contact, sexual harassment, or unwelcome sexual attention', 'Yes, Offensive comments related to gender, race, religion, or other protected characteristics', 'Yes, Disruption of talks or other events, or behavior interfering with participation', 'Yes, Inappropriate use of imagery, language, or personal attacks in virtual interactions")
]
###----------------------------------------------------------------------------------------------------###



###--- Language Model ---------------------------------------------------------------------------------###
model_llm_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
#model_llm_ID =  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
llm_tokenizer = AutoTokenizer.from_pretrained(model_llm_ID)
max_prompt_tokens = 14000   #16384 max
###----------------------------------------------------------------------------------------------------###



###----------------------------------------------------------------------------------------------------###
system_prompt = "You are a top-tier academic reviewer. You will be given a paper and asked structured questions one at a time. Answer with clarity and critical reasoning. No repetitions."
with open(PATH_output_prompt, "r", encoding="utf-8") as f:
    paper_prompt = f.read()
tokens = llm_tokenizer(paper_prompt, truncation=True, max_length=max_prompt_tokens, return_tensors="pt")
paper_prompt = llm_tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

previous_answers = []
final_review = []

for i, (header, question) in enumerate(review_questions, 1):
    print(f"\nQ{i}: {header}")

    if i == 1:
        full_prompt = f"{question}\n\n{paper_prompt}"
    else:
        full_prompt = f"{question}\n\n{paper_prompt}"
        #history = "\n".join(f"{q}\n{a}" for (q, a) in zip([q for q, _ in review_questions[:i-1]], previous_answers))
        #full_prompt = f"Review so far:\n{history}\n\nNow answer:\n{question}"

    response = client.chat.completions.create(
        model=model_llm_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.7,
        max_tokens=1024
    ).choices[0].message.content.strip()
    print(f"[LLM Answer - Q{i}]:\n{response}\n")

    response = re.sub(r"<think>.*?</think>", "", response, count=1, flags=re.DOTALL)

    # Normalize scores
    if "Score" in question or "rating" in question.lower():
        match = re.search(r"\b\d+\b", response)
        if match:
            score = int(match.group(0))
            response = f"{score}/10" if "10" in question else f"{min(score,5)}/5"

    final_review.append(f"{header} {response}")
    previous_answers.append(response)

with open(PATH_output_review, "w", encoding="utf-8") as f:
    for line in final_review:
        f.write(line + "\n")
print("Review written to:", PATH_output_review)
###----------------------------------------------------------------------------------------------------###

print("DONE.")