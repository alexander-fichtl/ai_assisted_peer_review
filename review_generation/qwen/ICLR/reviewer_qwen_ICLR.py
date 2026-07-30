"""
Input: Markdown produced by the Docling preprocessing pipeline (with ImageRefMode.REFERENCED)
- vLLM OpenAI-compatible server backend (Qwen3-VL)
- One pass per question over the whole paper context (selected sections + a limited amount of figures)
Outputs: per_question/*.json, final_review.json, final_review.html, audit.log
"""



#####----- Imports ----------------------------------------------------------------------------------------------------------#####
###--- Standard library
import base64
import copy
import gc
import json
import os
import re
import traceback
from io import BytesIO
from pathlib import Path
import requests
from typing import Any, Dict, List, Optional, Tuple

###--- Third-party
import numpy as np
from PIL import Image
import torch
import torch._dynamo
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Variables --------------------------------------------------------------------------------------------------------#####
CACHE_DIR = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
HF_TOKEN = os.environ.get("HF_TOKEN")

###--- Prompting
VAR_excludedSections = ["References", "Reference", "Acknowledgements", "Acknowledgement", "Author contributions", "Competing interests", "Additional information", "Appendix", "Supplementary", "Funding", "Corresponding authors", "Data availability", "Code availability"]
VAR_repeatPrevQA = True
VAR_fewshot = False
VAR_2ndPass = False
VAR_reviewerguide = True
VAR_areachairguide = True
VAR_codeofethics = True
VAR_codeofconduct = True
VAR_statistics = True

###--- Technical
torch._dynamo.config.capture_scalar_outputs = True
VAR_gpus = 4
VAR_quantized = False
VAR_maxContext = 64_000
VAR_tokensPerAnswer = 2000
VAR_tokensPerAnswer_2ndPass = 2000
VAR_model = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
VAR_setup = "vllm"
VAR_temperature = 0.6

###--- Server
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
VLLM_TIMEOUT = int(os.environ.get("VLLM_TIMEOUT", "600"))

###--- PATHS
PATH_output = Path("reviews")
PATH_input = Path("processed_pdfs_ICLR/processed_pdfs_ICLR")   #Docling preprocessing outputs (folder with JSON + images)
PATH_reviewerguide = Path("context/context_ICLR/reviewerguide.txt")
PATH_areachairguide = Path("context/context_ICLR/areachairguide.txt")
PATH_codeofethics = Path("context/context_ICLR/codeofethics.txt")
PATH_codeofconduct = Path("context/context_ICLR/codeofconduct.txt")
PATH_prevyear_stats = Path("context/context_ICLR/statistics_iclr24.txt")

###--- Images
VAR_maxImages = 5 #Set to 0 for ignoring images
VAR_maxImageSize = 4_000_000   # ~4MB
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Prompts ----------------------------------------------------------------------------------------------------------#####
PROMPT_system = (
    "You are a top-tier academic reviewer. Follow all rules:\
        1) Use only the provided context. Do not hallucinate.\
        2) No chain-of-thought in the final answer.\
        3) Write like a human reviewer: Be concise, relevant, specific, and non-repetitive. No boilerplate.\
        4) No restating of the questions or instructions.\
        5) Cite evidence from context. E.g. the section title or figure id or table id. You can also quote short sentence parts.\
        6) For rating/scoring questions (Soundness, Presentation, Contribution, Confidence, Overall): Reason briefly, then give the numeric rating in the form of 'Rating: <value>'.\
        7) For Strengths/Weaknesses/Suggestions prefer numbered lists (1., 2., …). Keep the amount of points reasonable. Only keep the least generic points.\
        8) You can incorporate Markdown and Latex into your review.\
        9) A total review has typically at least 300 and maximal 2000 words. Prefer your reviews to have between 500 and 1500 words."""
)

PROMPT_reviewQuestions = [
    ("Q1 (Summary):", "Briefly summarize the paper and its contributions. Do not critique the paper here. The authors should agree with a well-written summary. Typical range: 3-8 sentences."),
    ("Q2 (Soundness):", "Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence: 4 excellent, 3 good, 2 fair, 1 poor."),
    ("Q3 (Presentation):", "Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work: 4 excellent, 3 good, 2 fair, 1 poor."),
    ("Q4 (Contribution):", "Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader NeurIPS community? 4 excellent, 3 good, 2 fair, 1 poor."),
    ("Q5 (Strengths):", "Please give a substantive assessment of the strengths of the paper. Be specific, avoid generic remarks. This can, for example, include quality, clarity (e.g. well written), or significance, originality (e.g. a new definition or problem formulation, creative combinations of existing ideas, application to a new domain, or removing limitations from prior results.) If you have many, only list the least generic ones."),
    ("Q6 (Weaknesses):", "Please give a substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc. Unclarities are adressed later, so only list real weaknesses. If you have many, only list the least generic ones."),
    ("Q7 (Suggestions)", "Please list and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This is important for a productive rebuttal and discussion phase with the authors. Do not simply repeat all the weaknesses."),
    ("Q8 (Ethics):", "If there are ethical issues with this paper, please flag the paper for an ethics review and select area of expertise that the ethics reviewer should to have. Choose one of the following options: No ethics review needed, Ethics review needed: Discrimination / Bias / Fairness Concerns, Ethics review needed: Inadequate Data and Algorithm Evaluation, Ethics review needed: Inappropriate Potential Applications & Impact  (e.g., human rights concerns), Ethics review needed: Privacy and Security (e.g., consent, surveillance, data storage concern), Ethics review needed: Compliance (e.g., GDPR, copyright, license, terms of use), Ethics review needed: Research Integrity Issues (e.g., plagiarism), Ethics review needed: Responsible Research Practice (e.g., IRB, documentation, research ethics), Ethics review needed: Failure to comply with NeurIPS Code of Ethics (lack of required documentation, safeguards, disclosure, licenses, legal compliance)"),
    ("Q9 (Overall):", "Please provide an 'overall score' for the paper: 10: Award quality: Technically flawless paper with groundbreaking impact, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations. 9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI/ML and excellent impact on multiple areas of AI/ML, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations. 8: Strong Accept: Technically strong paper, with novel ideas, excellent impact on at least one area, or high-to-excellent impact on multiple areas, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations. 7: Accept: Technically solid paper, with high impact on at least one sub-area, or moderate-to-high impact on more than one areas, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations. 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations. 5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly. 4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly. 3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations. 2: Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations. 1: Very Strong Reject: For instance, a paper with trivial results or unaddressed ethical considerations."),
    ("Q10 (Confidence):", "Please provide a 'confidence score' for your assessment of this submission to indicate how confident you are in your evaluation. Give at most two brief sentences of justification, then exactly: 'Rating: <1–5>' with the following grounding: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully. 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked. 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked. 1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked."),
    ("Q11 (Conduct):", "If there are no violations of the Code of Conduct with this paper, please respond with NO. Otherwise, if this paper violates the Code of Conduct, please indicate the relevant section(s) from the following options: 'Yes, Harassment, bullying, or discrimination based on personal characteristics', 'Yes, Inappropriate physical contact, sexual harassment, or unwelcome sexual attention', 'Yes, Offensive comments related to gender, race, religion, or other protected characteristics', 'Yes, Disruption of talks or other events, or behavior interfering with participation', 'Yes, Inappropriate use of imagery, language, or personal attacks in virtual interactions")
]

PROMPT_fewshots = [
    ("not useful", """
     Summary: There have been many proposals in the recent literature to train fair ML models. This paper evaluates thousands of such models, and finds that a simple postprocessing technique achieves the fairness-accuracy Pareto frontier.
    
     Soundness: 4 excellent
     Presentation: 4 excellent
     Contribution: 4 excellent
     Strengths:
     This type of comprehensive benchmarking of thousands of models adds a ton of value to the algorithmic fairness literature. I think the result that a simple postprocessing step achieves the Pareo frontier is very significant. I applaud the authors for taking on this task.

     Weaknesses:
     None

     Questions:
     None

     Flag For Ethics Review: No ethics review needed.
     Rating: 8: accept, good paper
     Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
     Code Of Conduct: Yes"""),
    ("not useful", """
     Summary: Engaging in discussion on memorization within generative models, but set entirely against a synthetic toy data backdrop. The absence of explicit definitions for terms like inliers, outliers, and memorization stands out. It is a preliminary study and calls for diving deeper into the empirical analysis.

     Soundness: 1 poor
     Presentation: 1 poor
     Contribution: 1 poor
     Strengths:
     The topic of memorization in diffusion models is interesting and requires in depth study.

     Weaknesses:
     The claims and definitions are unclear.
     Experiments are carried out only on an arbitrary synthetic dataset
     Questions:
     NA

     Flag For Ethics Review: No ethics review needed.
     Rating: 1: strong reject
     Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
     Code Of Conduct: Yes
    """),
    ("not useful", """
     Summary: MetaGPT is a framework designed for programming in multi-agent collaborative environments, specifically utilizing the Large Language Model (LLM). Its main contribution is the integration of human workflows into LLM-based multi-agent collaboration, which is effective for complex software engineering tasks. The framework assigns different roles to the GPT model, similar to a pipeline in software development, including roles such as product manager, architect, and engineer. Its main advantage lies in generating coherent solutions and decomposing complex tasks through the collaboration of multiple AI agents.

     Soundness: 2 fair
     Presentation: 3 good
     Contribution: 2 fair
     
     Strengths:
     Problem-solving: MetaGPT can solve complex tasks by using different AI agents to handle different parts of the problem.
     Clear solutions: MetaGPT provides sensible solutions that favor tasks that require different elements to work together seamlessly.
     Flexibility: MetaGPT can be used for different tasks, and provide multiple outputs ranging from project plans to technical documentation.

     Weaknesses:
     Methodological innovation: A key issue with MetaGPT is the lack of innovation in its methodology. While it effectively utilizes Large Language Models (LLMs) in multi-agent systems, this approach may not be significantly different from existing approaches. Or MetaGPT is different from other methods in a trivial way, I don't really see their differences as being significant and requiring exploration with greater depth.
     Fairness of experimental comparisons: The comparison methods used to assess the effectiveness of MetaGPT do not appear to be fully equivalent to what MetaGPT offers. Such differences may lead to biased or misleading conclusions about their superiority or efficiency.
     Experimental validation of statements: Current experiments may not adequately validate the authors' claims about the efficiency and robustness of MetaGPT.

     Questions:
     My primary concern is that MetaGPT lacks methodological and theoretical innovation. To gain a more innovative edge, MetaGPT could deeply explore what deep-seated challenges in integrating newer AI techniques or unique collaboration strategies make it more clearly distinguishable from other LLM-based frameworks. (Honestly, I think MetaGPT is more suited for system demonstration conferences or tracks)
     In the Introduction, "MetaGPT stands out as a special solution that allows efficient meta-programming through a well-organized and specialized group of agents". How does the author define these professional concepts? For example, "efficient" and "well-organized".1) I haven't seen any evidence or convincing analysis of why it can be "efficient", nor have I seen experimental validation.2) How do you define "specialized group"? This seems trivial and can be varied.
     In the Introduction, "MetaGPT achieves a 100% task completion rate, further demonstrating the robustness and efficiency of our design", and "Our innovative integration of human-like SOPs throughout MetaGPT's design significantly enhances its robustness". 1) What is the definition of "robustness" here? I didn't see any discussion of "robustness" except in the Introduction section. 2) What are the deeper methodological challenges of integrating human-like SOPs?
     In the Related Work, "In this paper, we identify the main challenges in multi-agent cooperation: maintaining consistency, avoiding unproductive cycles, and controlling beneficial interactions". There seems to be inconsistency in the description of the research challenges between the "Related Work", the "Introduction" and the rest of the paper. Furthermore, is it trivial to maintain consistency in MetaGPT?" What are the professional definitions of "unproductive cycle" and "beneficial"? Are there experimental results to validate these advantages?
     In the Experiments, "We compare recent domain-specific LLMs in the code generation field ..." Is it fair to compare with these LLMs?
     In the Experiments, "We modified certain role-based prompts in order to instruct the MetaGPT framework to generate individual functions instead of entire classes" 1) Is it fair to compare with these frameworks? LangChain and AutoGPT, for example, which are not specific to the tasks involved in this commit. 2) How can we ensure that " prompt modification" does not lead to unfair comparisons?
     Figure 4. What are the results of multiple attempts and are they stable? Were there any problems, like robustness?
     In Section 4.2, what is "the cost of human revision" and how can it be rigorously measured?

     Flag For Ethics Review: Yes, Other reasons (please specify below)
     Details Of Ethics Concerns:
     This submission does not discuss any potential ethical issues, which I think is uncritical, especially for generative AI technologies. There are potential privacy, security, and bias issues that accompany the collection, processing, and use of data, exchange between agents, communication, and so on, in open-ended tasks with multi-person collaboration. For example, I perceive at least the following concerns and considerations.
     Unemployment and skill obsolescence: Automation of complex tasks in software engineering and other fields may lead to job losses. Professionals in these fields may need to adapt and acquire new skills to remain relevant. How to manage and communicate these changes is an ethical issue.
     Transparency and accountability: As AI takes on more complex collaborative tasks, it becomes critical to maintain transparency in decision-making. It should be clear how MetaGPT arrives at solutions and who is accountable for those decisions, especially in critical applications.
     Privacy and data security: The use of large-scale language models and AI in processing potentially sensitive information raises concerns about data privacy and security. Ensuring that user data is protected and used ethically is a key consideration.

     Rating: 3: reject, not good enough
     Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
     Code Of Conduct: Yes"""),
    ("useful", """
     Summary: This paper aims to give a formal justification for why truncation based sampling approaches work well in language generation. They link this phenomenon to the softmax bottleneck—the problem that the final linear layer in a neural network often bottlenecks the expressivity of the model. Explicitly, given the difference between the hidden (embedding) dimension and the vocabulary size, the final linear transformation before the softmax projection can only perform a low-rank projection. The authors claim that the resulting approximation to the target distribution is likely the source of model errors that leads to the “unreliable tail” probabilities observed by prior work. The authors develop an algorithm for uncovering which tokens are necessarily in the support of the target distribution, and propose to use this algorithm as the basis for a truncation sampling method. They provide empirical results (including human evaluations) when using this method.

     Soundness: 3 good
     Presentation: 4 excellent
     Contribution: 3 good
     
     Strengths:
     The work offers a theoretical explanation for why certain ad-hoc methods used during language generator decoding work well. This is a valuable insight to the NLG community
     The work then develops a sampling algorithm based on this theoretical explanation
     
     Weaknesses:
     The theoretical portion of the paper is at times difficult to understand due to notational choices and lack of specificity (for example, switching between individual token probabilities ). This is particularly important since the theoretical portion is the main contribution of the work
     The method does not appear to have empirical performance benefits and is computationally expensive, making it impractical
     There lacks robust empirical justification of the hypothesis. Figure 7, which is intended to show that including more of the original optimization problem constraints lead to better results, only consists of 3 points, which hardly feels like enough evidence to claim a “trend”.
     The terminology of the “true” distribution is perhaps misleading. I personally think that something like the “aggregate” distribution or the “data-generating” distribution would be more accurate
     A small point: the intro of section 4 has some grammatical errors
     
     Questions:
     Since the matrix W is static, can (3) not be solved for all elements of the vocabulary that meet the desired constraint ahead of time?
     Why is typical sampling omitted from the discussion of truncation-based sampling methods? It is arguably the most similar to the proposed method since it likewise “is able to discard higher-probability tokens while keeping… lower-probability tokens.” On a similar note, I don't understand footnote 3; I don't think it actually describes what is done by locally typical sampling
     The phrasing of footnote 1 is strange. Specifically, the statement “setting p∗ to be the 1-hot vector indicating the gold token” is underspecified. I imagine that this is referring to the conditional distribution for a particular prefix
     In theorem 1, these factors are the collective probability over/underestimation across all tokens, right? This then implies that no individual token probability can exceed these bounds. This logic should be made more explicit (the current notation is vague)
     I don't think that the softmax function satisfies the additivity property required of a “linear map.” Could you please elaborate on this claim at the top of page 6?
     On page 5, what is the concrete distinction between low vs. high quality tokens? Is this another way of saying in vs. out of the true distribution? It would be helpful to change the language here to align with the other terminology used by the paper
     The informal description about the practical implementation of basis-aware sampling is confusing. For example, what does “discarding the majority of the constraints” refer to?
     Given the observation that BAT sampling performs more strongly in lower entropy settings, a logical next step would be to see how it performs in translation or summarization, where historically, sampling has not led to the best results.
     It is perhaps more accurate to call the projection by W a linear layer instead of softmax layer, since the low-rank approximation is tied to this linear transformation, not the use of the softmax as a projection function. Further, are there insights into how the nature of these results will change with alternative projection functions, like the sparsemax?
     Can these principles be used to explain the degeneracy that happens when selecting high probability tokens, i.e., during greedy decoding?
     How do these results align with other work that has tried to explain why truncation methods work well in practice, such as [1]?
     
     [1] Meister et. al. 2023. On the Efficacy of Sampling Adapters.

     Flag For Ethics Review: No ethics review needed.
     Rating: 8: accept, good paper
     Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
     Code Of Conduct: Yes"""),
    ("useful", """
     Summary:
     The paper uses XAI methods for explaining factors that influence the moves of a chess engine. It claims two contributions:
     to replicate the results of McGrath et al. with an open source chess engine
     to propose a novel technique for computing salient squares. The proposed technique is trained to learn a binary mask for the features along with the model, similar to an attention mechanism (which the authors do not relate to, btw).
     
     Soundness: 1 poor
     Presentation: 3 good
     Contribution: 1 poor
    
     Strengths:
     The paper is well written and the approach is illustrated with good examples. The confirmation of the results of McGrath et al. on LeelaChess data is also valuable, but I don't think that this is a research contribution for this conference. It might be more interesting for a computer games conference.

     Weaknesses:
     Of the two claimed contributions, I don't think that 1. is actually a contribution, as there is no reason to expect that results with LeelaChess would be any different than the reported results for AlphaZero. The second contribution - the new method - is maybe more interesting. However, it does not seem to stick out from related work, some of which the authors seem to be unaware of (in particular the work on SARFA, which was published 3 years ago at this conference (Piyust et al. 2020), but also follow-up papers such as (Fritz & Fürnkranz 2021).
     The evidence for the validity of the proposed method is, as in previous work, only anecdotal, and I do not find the shown examples very convincing. The authors, e.g., claim that the examples shown in "...Fig. 5 capture the essence of what is necessary to solve the given puzzle....". Looking at these examples one sees that most (more than half) of the pieces are considered to be important in each of these positions, which includes pieces that are completely irrelevant for the assessment (e.g., Bd1 or Pb7 in Fig. 5 c)), and, on the other hand, miss pieces that are essential (such as Bd5 in Fig.5 c) which delivers the mate-in-1). The latter problem is briefly mentioned by the authors (but explained away), the former problem is not mentioned. Of course, if many pieces are annotated as important, there is a high chance that the essential pieces are among them. In more technical term, the proposed technique seems to have a high recall, but a rather low precision. Both of which are not evaluated.
     I think the paper would need a somewhat more founded evaluation methodology. For example, an explanation could be considered useful if it helps a human chess player to find the best move in the position. I doubt that any of the shown explanations can actually do that. Prior works, such as the above-mentioned work on SARFA, tried to at least get a grip on the accuracy of the evaluation by comparing the method's salient squares with those mentioned by human annotators. I am not sure that these techniques work better than what is proposed here, but at least the make a stronger attempt in evaluating it.
     Minor Comments:
     The authors often use "author (year)" citations in places where "(author year)" is clearly required, which is often quite confusing, such as "Monte Carlo Tree Search Coulom" in the first sentence of 2.2.
     There seems to be an error in Figure 3 (the pawn on the f-file is moved backwards by a square).
     I did not understand the idea behind formula (4). Why the max over all possible pieces for unoccupied squares?
     The related work saliency-based and perturbation-based methods are not suitable for chess. I did not understand this. The authors themselves propose a form of saliency maps, so I think they do not refer to the representation but to how it is computed in the 5 given references, but they do not explain. For perturbations, they do give an explanation, but it does not seem to be convincing to me (in particular, as the above-mentioned SARFA is based on perturbations).
    
     References: G. Piyush, P. Nikaash, V. Sukriti, K. Dhruv, D. Shripad, K. Balaji, and S. Sameer, “Explain your move: Understanding agent actions using specific and relevant feature attribution,” in International Conference on Learning Representations (ICLR), 2020. Jessica Fritz, Johannes Fürnkranz: Some Chess-Specific Improvements for Perturbation-Based Saliency Maps. CoG 2021: 1-8

     Questions:
     Can you say something about the precision of the squares annotated? As mentioned above, occupied squares tend to be blue, others tend to be red, so that the pieces that are involved in the target move tend to be blue, but many others as well. Are these really helpful for understanding the position? (e.g., those mentioned above, or Pb7 in Fig. 5 b), etc.).
     Your work also reminds me of attention mechanisms, maybe even transformer networks. Can you comment on that?

     Flag For Ethics Review: No ethics review needed.
     Rating: 3: reject, not good enough
     Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
     Code Of Conduct: Yes"""),
    ("useful", """
     Summary:
     This paper studies human and RL curriculum learning in a set of procedurally generated games.

     Soundness: 2 fair
     Presentation: 3 good
     Contribution: 1 poor
    
     Strengths:
     Parallel studies of human and machine learning is a very exciting line of work that will help make progress in both disciplines. I think the paper is well motivated and clearly written.

     Weaknesses:
     Level progress is a confusing term for curriculum learning researchers. “Learning progress” or “competence progress” is a standard proposition of intrinsic motivation to guide curricula, it refers to the derivative of the competence measure with respect to time: how much the competence increases or decreases. Here you use the term “progress” but it refers to what people call “competence,” a measure of success, a score. For instance, if you play the same level 10 times and reach the same score each time, there is no progress at all across-episodes, the progress is only intra-episode. I know that level progress describes what you expect it to describe in the language of video games, but it's confusing for researchers in this field: I thought you meant learning progress for the first 4-5 pages. Could you consider using another term? Competence is the standard term I think, score or performance could be good as well.
     The hypothesis supported here is not very clearly stated. I can see several alternative hypotheses that could explain the data:
     Children have some form of heuristic curriculum strategy that would look a bit like this: if i'm very good, i switch to a harder task, if i intermediate I stay there, if i'm bad i either stay or try something easier.
     Children have a causal understanding of the world and maximize their causal learning
     Children have an intermediate difficulty bias (~Florensa's paper): they keep doing the task for which they have intermediate competence / performance.
     Children have a bias towards learning progress: they can estimate expected learning progress in a model-based way and select the task that maximizes it. When a game is solved there is no further learning progress so they switch up, when they perform poorly they might expect more progress soon and stay, or not expect progress and recalibrate their estimations, which leads them to switch down.
     I feel like the paper is arguing for the second interpretation, although it's not stated clearly. I don't see anything in the paper that would allow us to argue for one more than the others?
     An interesting experiment could be to include levels from games different from the target game level. If children optimize for learning progress only, then we should see children select easy levels of the non-target games as well. Instead, we would probably see children almost never select the irrelevant game, which argues for a combination of intrinsic (LP-like) and extrinsic (going for the target level) curriculum.
     RL experiments:
     The switch threshold appears to be an important hyperparameter, it would be interesting to see how it affects the results: .9 seems high given that children seem to switch up around 75%.
     I'm not sure I understand the manual curriculum:
     why is there 16 parallel tasks?
     what does it mean to increment the difficulty by 1/16? I thought the difficulty was the number of lanes (1–5)? if the agent passes 0.9 in the current level (eg 1 lane), I expect the 16 tasks to become 2-lane tasks and the agent to be trained on these? Here it sounds like one of the 16 tasks becomes a lane-2 task, but then how is compute the score metric now? It it computed as the average level progress over the 16 taks (15 1-lane and 1 2-lane)?
     You use an on-policy algorithm that does not leverage a replay buffer. This means that old data from easier levels are thrown away. Using an off-policy algorithm (eg DDPG) would reuse past data, which may mitigate the catastrophic forgetting problem?
     Catastrophic forgetting for curriculum learning is a known problem, which is why all curriculum approaches perform stochastic selection: they do not switch from one level to the other but sample all levels with varying probabilities that are a function of the intrinsic motivation. Eg selecting the current best level with probability .7 and the rest with .3 / n_other_levels may fix the issue, see Jiang's paper and Colas 2019 for examples.
     Studies of curriculum algorithms always include the presentation of the random baseline selecting level uniformly. This baseline does not suffer from distribution shifts.
     I'm not sure how the intrinsic reward is used here. As far as I understand, it seems that the curriculum (level selection) is the same as before but that PPO now uses what the so-called intrinsic reward in addition, right? If so, this is a problem. Intrinsic motivations must be agnostic of the goal (this is what intrinsic means). They can be either state-based: assigning an intrinsic reward to states; or goal-based: assigning an intrinsic reward to goals / tasks. What you propose is to assign a reward to a state, but this reward is extrinsic, it measures performance in the task. Practically, you've just replaced a sparse reward with a dense reward, no intrinsic motivation here. What curriculum people do in that situation usually is to use that competence measure to guide the level-selection (goal-based reward): the level selected could be the one where there is the most progress (score now vs score before, not the level progress), or the one where the competence (level progress) is intermediate. These rewards are extrinsic because they are not tied to particular levels: the level-selector only cares about learning progress or intermediate difficulty, not about any level in particular. This would involve using a bandit algorithm to explore levels and maximize that score.
     The idea of a causal curriculum is interesting, but the papers cited when discussing this topic do not engage with any form of causality: eg Florensa 2018, Sukhbaatar 2017, Bengio 2009.
     A good way of discussing curriculum approaches is by making explicit the distal objective of the curriculum (maximizing performance on a set of target tasks), describing the proximal objective optimized by the approach (which usually includes forms of intrinsic motivation), and discussing what is the control parameter (the part of the MDP that is varied to maximize the proximal objective), see framework from Portelas et al 2020. Proximal objectives include: novelty, intermediate difficulty, learning progress, things that can be varied include: state space, transition space, goals, rewards, etc. The current review is not very clearly structured and omits large chunks of the field (eg learning progress maximizing methods). These are probably the closest to the one proposed in this paper so they should be mentioned. In particular, this paper proposes a curriculum over tasks (variation of state space, and transition function) where the proximal objective is an intermediate difficulty (intermediate level progress).
     What I would need to update my score:
     Further details and explanations for the points raised above
     Better account of existing curriculum approaches
     Explicit statement about the hypothesis proposed here + discussion about how the result support that hypothesis, and may or may not bring sufficient evidence to separate the different hypotheses i listed above
     I may have understood it wrong, but it seems that the authors completely missed the mark on the curriculum RL implementations: missing random baseline, deterministic level sampling instead of stochastic ones that's common in the field and, more importantly, the reward used is extrinsic and not intrinsic. It should be a goal-based reward and not a state-based reward. Again I may have understood it wrong so i'm open to discussions.
     I believe that this setup is interesting and the scientific goal is good. I'm looking forward to future improved versions of this paper.
     Minor comments:
     Vygotsky's citation is broken (missing y, no year).
     Questions:
     see above

     Flag For Ethics Review: No ethics review needed.
     Rating: 3: reject, not good enough
     Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
     Code Of Conduct: Yes""")
]

def build_system_prompt() -> str:
    system = PROMPT_system
    #Fewshot
    if VAR_fewshot and PROMPT_fewshots:
        lines = [
            "\n\nBelow are examples of review snippets from other papers. "
            "Mimic the style of 'useful' ones and avoid the style of 'not useful' ones. "
            "Use them only as guidance; do not copy them verbatim or the scores.",
        ]
        for label, ex in PROMPT_fewshots:
            lines.append(f"\n[{label.upper()}]\n{ex}")
        system = system + "\n" + "\n".join(lines)
    #Guidelines
    # Extra guidance text files (only if flags are True and files exist)
    if VAR_reviewerguide:
        txt = Path(PATH_reviewerguide).read_text(encoding="utf-8")
        system += "\n\n[Reviewer Guide]\n" + txt

    if VAR_areachairguide:
        txt = Path(PATH_areachairguide).read_text(encoding="utf-8")
        system += "\n\n[Area Chair Guide]\n" + txt

    if VAR_codeofethics:
        txt = Path(PATH_codeofethics).read_text(encoding="utf-8")
        system += "\n\n[Code of Ethics]\n" + txt

    if VAR_codeofconduct:
        txt = Path(PATH_codeofconduct).read_text(encoding="utf-8")
        system += "\n\n[Code of Conduct]\n" + txt
    if VAR_statistics:
            txt = PATH_prevyear_stats.read_text(encoding="utf-8")
            system += "\n\n[Previous Year Conference Statistics]\n" + txt

    return system

PROMPT_2ndPass_global = (
    "You are editing a complete academic peer review (answers to multiple fixed questions).\n"
    "Your task is to rewrite ALL answers to be much more concise and non-repetitive, while preserving meaning and factual content.\n"
    "Rules:\n"
    "1) Preserve every concrete technical point, criticism, suggestion, question, and positive remark that adds relevant information for the authors of the paper or the area chair.\n"
    "2) Remove filler, hedging, verbosity, and any sentences that only repeat earlier content.\n"
    "3) Eliminate duplication across answers: if a point appears in multiple answers, keep the most appropriate occurrence (e.g. repetitions between Weaknesses and Questions) and remove or shorten it elsewhere.\n"
    "4) Do NOT change any numeric ratings or options. Lines like 'Rating: <number>' and ethics / conduct labels must stay exactly as they are if present.\n"
    "5) Respect the role of each section: Summary is neutral description; Soundness/Presentation/Contribution briefly justify their ratings; Strengths and Weaknesses are numbered lists with only the most specific, non-generic items; Questions focus on unclarities or actionable requests.\n"
    "8) Output the edited review in exactly the same structure as the input: for each question, use the same heading line starting with '### ' and the same question header text, followed by its edited answer. Do not add or remove questions, and do not add commentary outside the review.\n"
)
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Parse preprocessed PDF -------------------------------------------------------------------------------------------#####
def image_to_dataUrl(path, max_bytes = VAR_maxImageSize):
    """Load image, re-encode to stay under max_bytes, return data URL or None if impossible."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            for quality in [95, 85, 75, 65, 50, 40, 30]:
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                data = buf.getvalue()
                if len(data) <= max_bytes:
                    b64 = base64.b64encode(data).decode("ascii")
                    return f"data:image/jpeg;base64,{b64}"
    except Exception:
        print(f"Error for image {path}. Skipped.")
        pass
    return None

def strip_thinking(s: str) -> str:
    """Remove <think>...</think> blocks (and variants) from model output."""
    LT = r"(?:&lt;|&#0*60;|&#x0*3c;)"
    GT = r"(?:&gt;|&#0*62;|&#x0*3e;)"
    # Cut everything from start through first closing tag
    for pat in [
        re.compile(r"(?is)\A.*?</\s*(think|thinking)\s*>"),
        re.compile(rf"(?is)\A.*?{LT}\s*/\s*(think|thinking)\s*{GT}"),
    ]:
        s, n = pat.subn("", s)
        if n:
            break
    # Remove any remaining full blocks
    for pat in [
        re.compile(r"(?is)<\s*(think|thinking)\s*>.*?<\s*/\s*\1\s*>"),
        re.compile(rf"(?is){LT}\s*(think|thinking)\s*{GT}.*?{LT}\s*/\s*\1\s*{GT}"),
        re.compile(r"(?is)```[ \t]*(think|thinking)[^\n]*\n.*?\n```"),
    ]:
        s = pat.sub("", s)
    # Remove stray tags
    for pat in [
        re.compile(r"(?is)<\s*/?(?:think|thinking)\s*>"),
        re.compile(rf"(?is){LT}\s*/?(?:think|thinking)\s*{GT}"),
    ]:
        s = pat.sub("", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()

def parse_md(md_path: Path, images_path: Path, exclude_sections: List[str],
             max_images: Optional[int] = None) -> Dict[str, Any]:
    """
    Docling-generated Markdown -> {"sections": [{"title","text","images":[{"path":...}]}]}
    Splits on ## headers, filters excluded sections, and collects image paths from
    the artifacts folder in order of appearance.
    """
    if not md_path.exists():
        raise FileNotFoundError(f"Missing preprocessed MD: {md_path}")

    text = md_path.read_text(encoding="utf-8")

    def is_excluded_title(title: str) -> bool:
        excl_norm = [
            ' '.join(((s or '').lower().strip().rstrip(':.')).split())
            for s in (exclude_sections or []) if s
        ]
        t = ' '.join(((title or '').lower().strip().rstrip(':.')).split())
        return any(
            t.startswith(e) or e.startswith(t)   # handles singular/plural both ways
            for e in excl_norm
        )

    # Split into blocks by any markdown header level
    header_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
    splits = list(header_pattern.finditer(text))

    out_sections: List[Dict[str, Any]] = []

    # Content before the first header goes into a default section
    if splits:
        pre = text[:splits[0].start()].strip()
        if pre:
            out_sections.append({"title": "Document", "text": pre, "images": []})
    else:
        # No headers at all — whole doc is one section
        out_sections.append({"title": "Document", "text": text.strip(), "images": []})
        return {"sections": out_sections}

    for i, match in enumerate(splits):
        title = match.group(1).strip()
        if is_excluded_title(title):
            continue
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        body = text[start:end].strip()

        # Extract image references from the body: ![...](path)
        img_pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
        images = []
        for img_match in img_pattern.finditer(body):
            raw_path = img_match.group(1).strip().replace("\\", "/")
            cand = (md_path.parent / raw_path).resolve()
            if cand.exists():
                images.append({"path": str(cand)})
            else:
                cand2 = images_path / Path(raw_path).name
                if cand2.exists():
                    images.append({"path": str(cand2)})

        # Remove image markdown syntax from body text so the LLM doesn't see raw paths
        clean_body = img_pattern.sub('', body).strip()

        out_sections.append({"title": title, "text": clean_body, "images": images})

    # Apply max_images limit across all sections
    if max_images is not None:
        total = 0
        for sec in out_sections:
            allowed = max(0, max_images - total)
            sec["images"] = sec["images"][:allowed]
            total += len(sec["images"])

    return {"sections": out_sections}
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- LLM Setup --------------------------------------------------------------------------------------------------------#####
def http_post(path: str, payload: dict) -> dict:
    url = VLLM_BASE_URL.rstrip("/") + path
    r = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=VLLM_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()

def load_model(modelID: str):
    """
    We are using an already-running vLLM OpenAI-compatible server (vllm serve ...),
    so 'load_model' just returns a lightweight handle/config.
    """
    return {"model": modelID}

def prompt_model(processor, llm, system: str, context_array, max_tokens: int, temperature: float):
    """
    Calls the vLLM OpenAI server at /v1/chat/completions.
    Builds OpenAI-style messages with image_url content (data URLs).
    """
    content = []
    loggable_content = []

    for x in context_array:
        if x.get("type") == "text":
            txt = x.get("text", "")
            content.append({"type": "text", "text": txt})
            loggable_content.append({"type": "text", "text": txt})

        elif x.get("type") == "image" and x.get("path"):
            path = x["path"]

            # vLLM OpenAI server accepts http(s) URLs or data: URLs.
            # Local file paths should be converted to data URLs.
            data_url = image_to_dataUrl(path, max_bytes=VAR_maxImageSize)
            if not data_url:
                print(f"[warn] Could not encode image under {VAR_maxImageSize} bytes: {path}. Skipped.")
                continue

            content.append({"type": "image_url", "image_url": {"url": data_url}})
            loggable_content.append({"type": "image", "path": path})

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    loggable_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": loggable_content},
    ]

    payload = {
        "model": (llm or {}).get("model", VAR_model),
        "messages": messages,
        "temperature": float(temperature) if temperature else 0.0,
        "max_tokens": int(max_tokens),
    }

    data = http_post("/v1/chat/completions", payload)
    out_text = (data["choices"][0]["message"]["content"] or "").strip()

    prompt_dump = {
        "messages": loggable_messages,
        "rendered_chat_template": None,  # server-side; we don't have it here
        "token_count": None,
        "max_new_tokens": int(max_tokens),
        "temperature": float(temperature) if temperature else 0.0,
        "raw_response": data,  # useful for debugging
    }

    return out_text, prompt_dump
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- HTML & JSON stuff ------------------------------------------------------------------------------------------------#####
def escape_html(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))

def write_html_report(path: Path, bundle: Dict[str, Any]) -> None:
    title = f"Review – {bundle.get('pdfname','')}"
    parts = [
        "<!doctype html>",
        "<html><head>",
        f"<meta charset='utf-8'><title>{escape_html(title)}</title>",
        "<style>",
        "body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5;margin:40px;color:#111}",
        "h1{font-size:1.6rem;margin-bottom:0.5rem}",
        "h2{font-size:1.1rem;margin-top:1.6rem;margin-bottom:0.3rem;color:#333}",
        ".q{border:1px solid #e5e7eb;border-radius:10px;padding:14px;margin:14px 0;background:#fafafa}",
        ".qh{font-weight:600;color:#111;margin-bottom:0.3rem}",
        ".qt{color:#444;margin-bottom:0.6rem}",
        ".ans{white-space:pre-wrap}",
        "</style>",
        "</head><body>",
        f"<h1>{escape_html(title)}</h1>",
    ]
    for q in bundle.get("questions", []):
        header = q["question_header"]
        question = q["question"]
        answer = q["final_answer"]
        parts += [
            "<div class='q'>",
            f"<div class='qh'>{escape_html(header)}</div>",
            f"<div class='qt'>{escape_html(question)}</div>",
            f"<div class='ans'>{escape_html(answer)}</div>",
            "</div>"
        ]
    parts += ["</body></html>"]
    path.write_text("\n".join(parts), encoding="utf-8")

def build_llm_jsons(pdfname: str,
                    q_records: List[Dict[str, Any]],
                    sections: List[Dict[str, Any]],
                    imgs_used: int,
                    model_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      html_json  -> {"pdfname": ..., "questions": q_records}  (drives your existing HTML)
      or_json    -> {"pdfname": ..., "content": {...}, "sections_used": [...], "images_used": N, "model": ...}
    """

    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = [r["final_answer"] for r in q_records[:11]]

    def rating_from_answer(ans: str, lo: int, hi: int):
        m = re.search(r"Rating:\s*([0-9]+)", ans or "", flags=re.I)
        if not m:
            return ans  #Fallback to raw text if format wrong
        v = int(m.group(1))
        return v if lo <= v <= hi else ans

    def ethics_list(ans: str) -> List[str]:
        t = (ans or "").strip()
        if not t:
            return []
        if re.search(r"\bno\s+ethics\b", t, flags=re.I):
            return ["No ethics review needed."]
        # split on ; or , if user listed multiple
        parts = [p.strip() for p in re.split(r"[;,]", t) if p.strip()]
        return parts if parts else [t]

    def conduct_value(ans: str) -> str:
        t = (ans or "").strip()
        if not t:
            return "NO"
        if re.fullmatch(r"no", t, flags=re.I) or re.search(r"\bno\b", t, flags=re.I):
            return "NO"
        return t

    ###--- OpenReview-like content block
    content = {
        "summary":                {"value": a1},
        "soundness":              {"value": rating_from_answer(a2, 1, 4)},
        "presentation":           {"value": rating_from_answer(a3, 1, 4)},
        "contribution":           {"value": rating_from_answer(a4, 1, 4)},
        "strengths":              {"value": a5},
        "weaknesses":             {"value": a6},
        "questions":              {"value": a7},
        "flag_for_ethics_review": {"value": ethics_list(a8)},
        "rating":                 {"value": rating_from_answer(a9, 1, 10)},
        "confidence":             {"value": rating_from_answer(a10, 1, 5)},
        "code_of_conduct":        {"value": conduct_value(a11)},
    }

    html_json = {
        "pdfname": pdfname,
        "questions": q_records,
    }

    or_json = {
        "pdfname": pdfname,
        "content": content,
        "sections_used": [s["title"] for s in sections],
        "images_used": imgs_used,
        "model": model_id,
    }

    return html_json, or_json
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- 2nd Pass ---------------------------------------------------------------------------------------------------------#####
def split_headings_block(text: str) -> Dict[str, str]:
    """
    Given a markdown block:

        ### Q1 (Summary):
        answer...

        ### Q2 (Soundness):
        answer...

    return a dict: { "Q1 (Summary):": "answer...", "Q2 (Soundness):": "answer..." }.
    """
    lines = text.splitlines()
    chunks: Dict[str, List[str]] = {}
    current_header: Optional[str] = None
    buf: List[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        m = re.match(r"^###\s+(Q[0-9]+\s*\([^)]*\):?)\s*$", line.strip())
        if m:
            # flush previous
            if current_header is not None:
                chunks[current_header] = "\n".join(buf).strip()
                buf = []
            current_header = m.group(1)
        else:
            if current_header is not None:
                buf.append(raw_line)

    if current_header is not None:
        chunks[current_header] = "\n".join(buf).strip()

    return {k: v for k, v in chunks.items() if v}

def tighten_full_review(
    processor,
    model,
    q_records: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Second pass over the complete review: aggressively compress,
    remove redundancy within and across answers,
    but keep ratings/flags and structure.
    """
    if not q_records:
        return q_records, {}

    # Build one markdown block with all answers
    parts: List[str] = []
    for rec in q_records:
        header = rec["question_header"]
        answer = rec["final_answer"] or ""
        parts.append(f"### {header}\n{answer.strip()}\n")

    full_review_text = "\n".join(parts).strip()

    context_text = (
        "Edit the following complete review.\n"
        "Make each answer much more concise and remove repetition both within answers "
        "and across different answers, following your rules.\n\n"
        "REVIEW START\n"
        f"{full_review_text}\n"
        "REVIEW END"
    )

    context = [{"type": "text", "text": context_text}]

    edited_text, prompt_dump = prompt_model(
        processor=processor,
        llm=model,
        system=PROMPT_2ndPass_global,
        context_array=context,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    edited_text = (edited_text or "").strip()
    if not edited_text:
        # If something went wrong, keep originals.
        return q_records, prompt_dump

    # Parse edited answers back into per-question chunks
    edited_chunks = split_headings_block(edited_text)

    for rec in q_records:
        header = rec["question_header"]
        original = rec.get("final_answer", "")
        rec["raw_answer"] = original  # keep original
        new = edited_chunks.get(header)
        if new:
            rec["final_answer"] = new.strip()

    return q_records, prompt_dump

def snapshot_rating_values(q_records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    For each question_header, collect the list of numeric rating values (as strings)
    found in lines like 'Rating: <number>' in the first-pass answers.
    """
    pat = re.compile(r"Rating:\s*([0-9]+)", re.I)
    snapshot: Dict[str, List[str]] = {}
    for rec in q_records:
        header = rec.get("question_header", "")
        ans = rec.get("final_answer") or ""
        vals = pat.findall(ans)
        snapshot[header] = sorted(set(vals))
    return snapshot

def ratings_preserved(
    original_snapshot: Dict[str, List[str]],
    new_q_records: List[Dict[str, Any]],
) -> bool:
    """
    Check that for every question_header, the multiset of rating values (as numbers)
    is identical before and after tightening. If any rating disappears, changes,
    or new ratings appear where there were none, return False.
    """
    pat = re.compile(r"Rating:\s*([0-9]+)", re.I)
    for rec in new_q_records:
        header = rec.get("question_header", "")
        ans = rec.get("final_answer") or ""
        orig_vals = original_snapshot.get(header, [])
        new_vals = sorted(set(pat.findall(ans)))
        if orig_vals != new_vals:
            return False

    # Also ensure no header disappeared entirely
    new_headers = {rec.get("question_header", "") for rec in new_q_records}
    for header in original_snapshot.keys():
        if header not in new_headers:
            return False

    return True
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Main -------------------------------------------------------------------------------------------------------------#####
def run_review(pdfname: str, processor=None, model=None):
    debug = True
    outdir = PATH_output / pdfname
    outdir.mkdir(parents=True, exist_ok=True)
    mdpath = PATH_input / pdfname / f"{pdfname}.md"
    ipath  = PATH_input / pdfname / f"{pdfname}_artifacts"

    parsed = parse_md(mdpath, ipath, VAR_excludedSections, max_images=VAR_maxImages if VAR_maxImages > 0 else None)
    if debug:
        print(f"\nParsed json:\n\n{parsed}")

    #Build context (sections + images)
    context_base = []
    sections = parsed["sections"]
    imgs_used = 0
    images_total = sum(len(s.get("images", [])) for s in sections)
    warned = False

    for s in sections:
        context_base.append({"type": "text", "text": f"# {s['title']}\n{s['text']}"})
        for im in s.get("images", []):
            context_base.append({"type": "image", "path": im["path"]})
            imgs_used += 1

    if VAR_maxImages > 0 and imgs_used == VAR_maxImages:
        print(f"[info] Image limit reached: including {imgs_used} of {images_total} total images.")

    if model is None:
        model = load_model(VAR_model)

    system_prompt = build_system_prompt() 
    prev_qa = []
    q_records = []

    with (outdir / "audit.log").open("a", encoding="utf-8") as audit:
        for idx, (header, question) in enumerate(PROMPT_reviewQuestions, 1):
            context = list(context_base)

            #Current question
            context.append({"type": "text", "text": f"\n\nYou will have to answer this question:\n{question}"})

            #Add previous Q/A 
            if VAR_repeatPrevQA and prev_qa:
                hist_text = "\n".join([f"{h}\nQ: {q}\nA (summary): {a[:500]}" for h, q, a in prev_qa])
                context.append({"type": "text", "text": f"\n\nPrevious Q/A:\n{hist_text}"})

            #Append question
            context.append({"type": "text", "text": f"\n\nNow answer this question:\n{question}"})

            answer_raw, prompt_dump = prompt_model(processor, model, system_prompt, context, VAR_tokensPerAnswer, VAR_temperature)
            answer = strip_thinking(answer_raw)

            qpath_thinking = outdir / "per_question" / f"q{idx:02d}_thinking.txt"
            qpath_thinking.parent.mkdir(parents=True, exist_ok=True)
            qpath_thinking.write_text(answer_raw, encoding="utf-8")

            rec = {
                "question_header": header,
                "question": question,
                "final_answer": answer,
                "sections_used": [s["title"] for s in sections],
                "images_used": imgs_used,
                "prompt": {
                    "system": system_prompt, #PROMPT_system,
                    "messages": prompt_dump.get("messages"),                 # images as file paths
                    "rendered_chat_template": prompt_dump.get("rendered_chat_template"),
                    "token_count": prompt_dump.get("token_count"),
                    "max_new_tokens": prompt_dump.get("max_new_tokens"),
                    "temperature": prompt_dump.get("temperature"),
                },
                # Optional: the exact user payload you constructed
                "context_for_llm": context,  # text + image paths (JSON-serializable)
            }

            qpath = outdir / "per_question" / f"q{idx:02d}.json"
            qpath.parent.mkdir(parents=True, exist_ok=True)
            qpath.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

            #q_records.append(rec)
            q_records.append({k: rec[k] for k in ["question_header","question","final_answer","sections_used","images_used"]})
            prev_qa.append((header, question, answer))
            audit.write(f"Q{idx:02d}: max_new_tokens={VAR_tokensPerAnswer}, images={imgs_used}\n")
                    
        first_pass_q_records = copy.deepcopy(q_records)
        # Snapshot ratings before 2nd pass, so we can verify they don't change
        original_rating_snapshot = snapshot_rating_values(first_pass_q_records)

        tighten_prompt_dump = None
        if VAR_2ndPass:
            try:
                q_records, tighten_prompt_dump = tighten_full_review(
                    processor=processor,
                    model=model,
                    q_records=q_records,
                    max_tokens=VAR_tokensPerAnswer_2ndPass,
                    temperature=VAR_temperature,
                )
                # Optional: log tightening metadata
                if tighten_prompt_dump:
                    (outdir / "tighten_review_prompt.json").write_text(
                        json.dumps(tighten_prompt_dump, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                ##-- ratings safety check
                if not ratings_preserved(original_rating_snapshot, q_records):
                    print(
                        f"[warn] 2nd pass changed ratings for {pdfname}; "
                        "reverting to 1st-pass review."
                    )
                    q_records = first_pass_q_records

            except Exception as e:
                print(f"[warn] tightening full review failed for {pdfname}: {e}")
                q_records = first_pass_q_records

    html_json, or_json = build_llm_jsons(
        pdfname=pdfname,
        q_records=q_records,
        sections=sections,
        imgs_used=imgs_used,
        model_id=VAR_model,
    )

    (outdir / "final_review.json").write_text(
        json.dumps(html_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_html_report(outdir / "final_review.html", html_json)

    #OpenReview-like JSON
    (outdir / "final_review_openreview.json").write_text(
        json.dumps(or_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ##-- If 2nd pass was used, also write 1st-pass versions 
    if VAR_2ndPass:
        html_json_1stPass, or_json_1stPass = build_llm_jsons(
            pdfname=pdfname,
            q_records=first_pass_q_records,  # pre-tightening answers
            sections=sections,
            imgs_used=imgs_used,
            model_id=VAR_model,
        )

        (outdir / "final_review_1stPass.json").write_text(
            json.dumps(html_json_1stPass, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_html_report(outdir / "final_review_1stPass.html", html_json_1stPass)

        (outdir / "final_review_openreview_1stPass.json").write_text(
            json.dumps(or_json_1stPass, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


    print(f"Wrote: {outdir/'final_review.json'}, {outdir/'final_review_openreview.json'} and {outdir/'final_review.html'}")
#####------------------------------------------------------------------------------------------------------------------------#####


if __name__ == "__main__":
    processor = None
    model = load_model(VAR_model)
    pdfnames = ["gL2plIZ4Sf", "CWpQsAubxy", "owFVvaLFdw", "IGcGUHEkM5", "mpAMH1OyMO", "7y11BdJIOp", "rv0j1CyRQK", "KMFotFeGjJ", "gxkOfdaaMA", "ooYtHcj6LI", "w0rVXs6QJM", "EFOMez9cBW", "kVojSLUcvS", "liDUSnwUVP", "Obefq4k8iG", "sDlIWuHaRt", "WNHNfkulYA", "8uZ5UdIul2", "koIbbsfKSf", "gpeXJ9UI07", "xmsbFPmsOz", "6K08FPo2cf", "WrEQFwWCdT", "084SvT55yk", "ugGIpVfGE6", "Ba0RlUn0TR", "lAlI5FuIf7", "BZ1SSMS7L6", "KfjpyOcPQj", "aQia5HdgBY", "jKcHPxRvk9", "Ovmvu3IZAp", "jfhIbJ3K8e", "ifC1qxP5rS", "dOxOX4GF1k", "OWHKdYwYiF", "IljH2Bt6jO", "Hgw56DUFzD", "CYmjrbQRyM", "zZNYUkBS77", "GP5RHZnEsw", "KJ0ktHAuds", "rPrNaDJrLx", "PS1YS8Wv4t", "1RrwTXSU4c", "fdpB9RBFv5", "5IqvMNQaUg", "XCW1l9qcxy", "nBta7psl5J", "dc90uPqxWF"]

    for pdfname in pdfnames:
        try:
            print(f"\nRunning review for {pdfname}.")
            run_review(pdfname=pdfname, processor=processor, model=model)
        except torch.cuda.OutOfMemoryError as e:
            # log and recover VRAM, then continue with next paper
            outdir = PATH_output / pdfname
            outdir.mkdir(parents=True, exist_ok=True)
            msg = f"[OOM] {pdfname}: {e}\n{traceback.format_exc()}"
            print(msg)
            (outdir / "error.log").write_text(msg, encoding="utf-8")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            # log generic error and continue
            outdir = PATH_output / pdfname
            outdir.mkdir(parents=True, exist_ok=True)
            msg = f"[error] {pdfname}: {e}\n{traceback.format_exc()}"
            print(msg)
            (outdir / "error.log").write_text(msg, encoding="utf-8")
            continue
    print("\nDONE.")