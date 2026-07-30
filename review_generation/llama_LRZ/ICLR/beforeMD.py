"""
Input: JSON produced by the Docling preprocessing pipeline (document.save_as_json with ImageRefMode.REFERENCED)
- Local Transformers backend (AutoProcessor + Llama4ForConditionalGeneration)
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
from typing import Any, Dict, List, Optional, Tuple

###--- Third-party
import numpy as np
from PIL import Image
import torch
import torch._dynamo
from transformers import AutoProcessor, Llama4ForConditionalGeneration
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- Variables --------------------------------------------------------------------------------------------------------#####
CACHE_DIR = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
HF_TOKEN = os.environ.get("HF_TOKEN")

###--- Prompting
VAR_excludedSections = ["References", "Acknowledgements", "Acknowledgments", "Appendix", "Supplementary"]
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
VAR_model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
VAR_setup = "transformers" #vLLM #quantized 
VAR_temperature = 0.0

###--- PATHS
PATH_output = Path("reviews")
PATH_input = Path("pdf_preprocessing/processed_pdfs")   #Docling preprocessing outputs (folder with JSON + images)
PATH_reviewerguide = Path("context/reviewerguide.txt")
PATH_areachairguide = Path("context/areachairguide.txt")
PATH_codeofethics = Path("context/codeofethics.txt")
PATH_codeofconduct = Path("context/codeofconduct.txt")
PATH_prevyear_stats = Path("context/statistics_iclr24.txt")

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
        3) Cite evidence from context. E.g. the section title or figure id or table id. You can also quote short sentence parts.\
        4) For rating/scoring questions (Soundness, Presentation, Contribution, Confidence, Overall): Reason briefly, then give the numeric rating in the form of 'Rating: <value>'.\
        5) For Strengths/Weaknesses/Suggestions prefer numbered lists (1., 2., …). Keep the amount of points reasonable. Only keep the least generic points.\
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
    ("Q9 (Overall):", "Please provide an overall score for the paper on the 1–10 scale. Use the full scale, not just 7–8: most submissions should fall between 3 and 7, 8–10 are reserved for clearly outstanding work, and 1–3 for clearly flawed or low-impact work. Your score must be consistent with your earlier ratings for Soundness, Presentation and Contribution and with the severity and number of weaknesses. Give at most 3 concise sentences of justification and then, on a separate line, output exactly: 'Rating: <number>'. This is the scale: 10: Award quality: Technically flawless paper with groundbreaking impact, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations. 9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI/ML and excellent impact on multiple areas of AI/ML, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations. 8: Strong Accept: Technically strong paper, with novel ideas, excellent impact on at least one area, or high-to-excellent impact on multiple areas, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations. 7: Accept: Technically solid paper, with high impact on at least one sub-area, or moderate-to-high impact on more than one areas, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations. 6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations. 5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly. 4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly. 3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations. 2: Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations. 1: Very Strong Reject: For instance, a paper with trivial results or unaddressed ethical considerations."),
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
     In the Introduction, "MetaGPT achieves a 100% task completion rate, further demonstrating the robustness and efficiency of our design", and "Our innovative integration of human-like SOPs throughout MetaGPT’s design significantly enhances its robustness". 1) What is the definition of "robustness" here? I didn't see any discussion of "robustness" except in the Introduction section. 2) What are the deeper methodological challenges of integrating human-like SOPs?
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
     Why is typical sampling omitted from the discussion of truncation-based sampling methods? It is arguably the most similar to the proposed method since it likewise “is able to discard higher-probability tokens while keeping… lower-probability tokens.” On a similar note, I don’t understand footnote 3; I don’t think it actually describes what is done by locally typical sampling
     The phrasing of footnote 1 is strange. Specifically, the statement “setting p∗ to be the 1-hot vector indicating the gold token” is underspecified. I imagine that this is referring to the conditional distribution for a particular prefix
     In theorem 1, these factors are the collective probability over/underestimation across all tokens, right? This then implies that no individual token probability can exceed these bounds. This logic should be made more explicit (the current notation is vague)
     I don’t think that the softmax function satisfies the additivity property required of a “linear map.” Could you please elaborate on this claim at the top of page 6?
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
     Level progress is a confusing term for curriculum learning researchers. “Learning progress” or “competence progress” is a standard proposition of intrinsic motivation to guide curricula, it refers to the derivative of the competence measure with respect to time: how much the competence increases or decreases. Here you use the term “progress” but it refers to what people call “competence,” a measure of success, a score. For instance, if you play the same level 10 times and reach the same score each time, there is no progress at all across-episodes, the progress is only intra-episode. I know that level progress describes what you expect it to describe in the language of video games, but it’s confusing for researchers in this field: I thought you meant learning progress for the first 4-5 pages. Could you consider using another term? Competence is the standard term I think, score or performance could be good as well.
     The hypothesis supported here is not very clearly stated. I can see several alternative hypotheses that could explain the data:
     Children have some form of heuristic curriculum strategy that would look a bit like this: if i’m very good, i switch to a harder task, if i intermediate I stay there, if i’m bad i either stay or try something easier.
     Children have a causal understanding of the world and maximize their causal learning
     Children have an intermediate difficulty bias (~Florensa’s paper): they keep doing the task for which they have intermediate competence / performance.
     Children have a bias towards learning progress: they can estimate expected learning progress in a model-based way and select the task that maximizes it. When a game is solved there is no further learning progress so they switch up, when they perform poorly they might expect more progress soon and stay, or not expect progress and recalibrate their estimations, which leads them to switch down.
     I feel like the paper is arguing for the second interpretation, although it’s not stated clearly. I don’t see anything in the paper that would allow us to argue for one more than the others?
     An interesting experiment could be to include levels from games different from the target game level. If children optimize for learning progress only, then we should see children select easy levels of the non-target games as well. Instead, we would probably see children almost never select the irrelevant game, which argues for a combination of intrinsic (LP-like) and extrinsic (going for the target level) curriculum.
     RL experiments:
     The switch threshold appears to be an important hyperparameter, it would be interesting to see how it affects the results: .9 seems high given that children seem to switch up around 75%.
     I’m not sure I understand the manual curriculum:
     why is there 16 parallel tasks?
     what does it mean to increment the difficulty by 1/16? I thought the difficulty was the number of lanes (1–5)? if the agent passes 0.9 in the current level (eg 1 lane), I expect the 16 tasks to become 2-lane tasks and the agent to be trained on these? Here it sounds like one of the 16 tasks becomes a lane-2 task, but then how is compute the score metric now? It it computed as the average level progress over the 16 taks (15 1-lane and 1 2-lane)?
     You use an on-policy algorithm that does not leverage a replay buffer. This means that old data from easier levels are thrown away. Using an off-policy algorithm (eg DDPG) would reuse past data, which may mitigate the catastrophic forgetting problem?
     Catastrophic forgetting for curriculum learning is a known problem, which is why all curriculum approaches perform stochastic selection: they do not switch from one level to the other but sample all levels with varying probabilities that are a function of the intrinsic motivation. Eg selecting the current best level with probability .7 and the rest with .3 / n_other_levels may fix the issue, see Jiang’s paper and Colas 2019 for examples.
     Studies of curriculum algorithms always include the presentation of the random baseline selecting level uniformly. This baseline does not suffer from distribution shifts.
     I’m not sure how the intrinsic reward is used here. As far as I understand, it seems that the curriculum (level selection) is the same as before but that PPO now uses what the so-called intrinsic reward in addition, right? If so, this is a problem. Intrinsic motivations must be agnostic of the goal (this is what intrinsic means). They can be either state-based: assigning an intrinsic reward to states; or goal-based: assigning an intrinsic reward to goals / tasks. What you propose is to assign a reward to a state, but this reward is extrinsic, it measures performance in the task. Practically, you’ve just replaced a sparse reward with a dense reward, no intrinsic motivation here. What curriculum people do in that situation usually is to use that competence measure to guide the level-selection (goal-based reward): the level selected could be the one where there is the most progress (score now vs score before, not the level progress), or the one where the competence (level progress) is intermediate. These rewards are extrinsic because they are not tied to particular levels: the level-selector only cares about learning progress or intermediate difficulty, not about any level in particular. This would involve using a bandit algorithm to explore levels and maximize that score.
     The idea of a causal curriculum is interesting, but the papers cited when discussing this topic do not engage with any form of causality: eg Florensa 2018, Sukhbaatar 2017, Bengio 2009.
     A good way of discussing curriculum approaches is by making explicit the distal objective of the curriculum (maximizing performance on a set of target tasks), describing the proximal objective optimized by the approach (which usually includes forms of intrinsic motivation), and discussing what is the control parameter (the part of the MDP that is varied to maximize the proximal objective), see framework from Portelas et al 2020. Proximal objectives include: novelty, intermediate difficulty, learning progress, things that can be varied include: state space, transition space, goals, rewards, etc. The current review is not very clearly structured and omits large chunks of the field (eg learning progress maximizing methods). These are probably the closest to the one proposed in this paper so they should be mentioned. In particular, this paper proposes a curriculum over tasks (variation of state space, and transition function) where the proximal objective is an intermediate difficulty (intermediate level progress).
     What I would need to update my score:
     Further details and explanations for the points raised above
     Better account of existing curriculum approaches
     Explicit statement about the hypothesis proposed here + discussion about how the result support that hypothesis, and may or may not bring sufficient evidence to separate the different hypotheses i listed above
     I may have understood it wrong, but it seems that the authors completely missed the mark on the curriculum RL implementations: missing random baseline, deterministic level sampling instead of stochastic ones that's common in the field and, more importantly, the reward used is extrinsic and not intrinsic. It should be a goal-based reward and not a state-based reward. Again I may have understood it wrong so i'm open to discussions.
     I believe that this setup is interesting and the scientific goal is good. I'm looking forward to future improved versions of this paper.
     Minor comments:
     Vygotsky’s citation is broken (missing y, no year).
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

def parse_json(json_path: Path, images_path: Path, exclude_sections: List[str]) -> Dict[str, Any]:
    """
    Docling JSON -> {"sections": [{"title","text","images":[{"path":...}]}]}
    Groups content by section headers. Excludes sections whose title starts with any of exclude_sections (case-insensitive).
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Missing preprocessed JSON: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    ###--- Helper functions
    def norm(s):
        return s.strip() if isinstance(s, str) else s

    def is_excluded_title(title: str) -> bool:
        excl_norm = tuple(
            ' '.join(((s or '').lower().strip().rstrip(':.')).split())
            for s in (exclude_sections or []) if s
        )
        t = ' '.join(((title or '').lower().strip().rstrip(':.')).split())
        return bool(excl_norm) and t.startswith(excl_norm)

    def resolve_ref(node: Any) -> Any:
        if isinstance(node, dict) and "$ref" in node:
            ref = norm(node.get("$ref"))
            return IDX.get(ref)
        return node
    
    def index_block(name: str):
        arr = data.get(name) or []
        return {f"#/{name}/{i}": obj for i, obj in enumerate(arr)}

    IDX = {}
    for k in ("texts", "pictures", "tables", "groups", "pages"):
        IDX.update(index_block(k))

    ###--- Image stuff
    def _save_data_uri_to_file(data_uri: str, out_dir: Path, stem: str, hint: str = "") -> str | None:
        data_uri = norm(data_uri)
        m = re.match(r"^data:(image/(png|jpeg|jpg));base64,(.+)$", data_uri or "", flags=re.IGNORECASE)
        if not m:
            return None
        ext = ".png" if m.group(2).lower() == "png" else ".jpg"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{stem}{('_' + hint) if hint else ''}{ext}"
        path = out_dir / fname
        try:
            path.write_bytes(base64.b64decode(m.group(3)))
            return str(path)
        except Exception:
            return None

    def _resolve_image_uri(uri: str) -> str | None:
        uri = norm(uri)
        if not isinstance(uri, str):
            return None
        if uri.startswith("data:image/"):
            return _save_data_uri_to_file(uri, images_path, json_path.stem)

        u = uri.replace("\\", "/")
        p = Path(u)

        if p.is_absolute() and p.exists():
            return str(p)

        cand = (json_path.parent / p).resolve()
        if cand.exists():
            return str(cand)

        cand2 = (images_path / p.name).resolve()
        if cand2.exists():
            return str(cand2)

        return None
    
    ###--- Table stuff
    def _cell_text(x):
        if isinstance(x, dict):
            return (x.get("text") or x.get("orig") or "").strip()
        if isinstance(x, str):
            return x.strip()
        return ""

    def _grid_to_tsv(grid):
        if not grid or not grid[0]:
            return None
        width = len(grid[0])
        lines = []
        for r in grid:
            row = (r + [""] * width)[:width]   # normalize ragged rows
            # keep it single-line per row; tabs between cells
            lines.append("\t".join(c.replace("\n", " ").strip() for c in row))
        return "\n".join(lines)

    def _docling_table_to_tsv(tbl):
        """
        Supports DoclingDocument tables like:
        tbl["data"]["grid"] -> 2D array of cell dicts (preferred)
        tbl["data"]["table_cells"] -> flat list w/ start/end offsets + spans (fallback)
        Also keeps legacy formats:
        A) {"rows": [[...], ...]}
        B) {"rows": [{"cells":[...]} , ...]}
        C) {"cells":[{"row":i,"col":j,...}, ...]}
        Return TSV string or None.
        """
        if not isinstance(tbl, dict):
            return None

        # ---- D) DoclingDocument v1.x: data.grid
        data_ = tbl.get("data")
        if isinstance(data_, dict):
            grid = data_.get("grid")
            if isinstance(grid, list) and grid and isinstance(grid[0], list):
                matrix = [[_cell_text(cell) for cell in row] for row in grid]
                return _grid_to_tsv(matrix)

            # ---- D2) DoclingDocument: data.table_cells
            cells = data_.get("table_cells")
            if isinstance(cells, list) and cells and isinstance(cells[0], dict):
                def _i(x, default=0):
                    try:
                        return int(x)
                    except Exception:
                        return default

                max_r = 0
                max_c = 0
                for c in cells:
                    sr = _i(c.get("start_row_offset_idx", c.get("row", 0)))
                    er = _i(c.get("end_row_offset_idx", sr))
                    sc = _i(c.get("start_col_offset_idx", c.get("col", 0)))
                    ec = _i(c.get("end_col_offset_idx", sc))
                    max_r = max(max_r, er)
                    max_c = max(max_c, ec)

                matrix = [["" for _ in range(max_c + 1)] for _ in range(max_r + 1)]

                for c in cells:
                    sr = _i(c.get("start_row_offset_idx", c.get("row", 0)))
                    sc = _i(c.get("start_col_offset_idx", c.get("col", 0)))
                    # anchor-only: prevents repeating text across merged spans
                    matrix[sr][sc] = _cell_text(c)

                return _grid_to_tsv(matrix)

        # ---- A) rows as list-of-lists
        rows = tbl.get("rows")
        if isinstance(rows, list) and rows and isinstance(rows[0], list):
            grid = [[_cell_text(c) for c in r] for r in rows]
            return _grid_to_tsv(grid)

        # ---- B) rows as list of objects with "cells"
        if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "cells" in rows[0]:
            grid = [[_cell_text(c) for c in (r.get("cells") or [])] for r in rows]
            return _grid_to_tsv(grid)

        # ---- C) sparse cells with (row,col)
        cells = tbl.get("cells")
        if isinstance(cells, list) and cells and isinstance(cells[0], dict):
            max_r = max(int(c.get("row", 0) or 0) for c in cells)
            max_c = max(int(c.get("col", 0) or 0) for c in cells)
            grid = [["" for _ in range(max_c + 1)] for _ in range(max_r + 1)]
            for c in cells:
                r = int(c.get("row", 0) or 0)
                k = int(c.get("col", 0) or 0)
                grid[r][k] = _cell_text(c)
            return _grid_to_tsv(grid)

        return None

    ###--- Parsing
    body = data.get("body") or {}
    children = [resolve_ref(ch) for ch in (body.get("children") or [])]

    out_sections: List[Dict[str, Any]] = []
    cur = None
    seen_any_header = False
    in_excluded = False
    pending_title = None

    def _new_section(title: str):
        nonlocal out_sections, in_excluded, seen_any_header
        title = norm(title) or "(untitled)"
        seen_any_header = True
        if is_excluded_title(title):
            in_excluded = True
            return None
        in_excluded = False
        sec = {"title": title, "text": "", "images": []}
        out_sections.append(sec)
        return sec

    for node in children:
        if not isinstance(node, dict):
            continue

        raw_label = node.get("label")
        label = norm(raw_label)
        label_lc = (label or "").lower()

        text = norm(node.get("text") or node.get("orig") or "")
        self_ref = norm(node.get("self_ref", ""))

        #Headers start new sections
        if label_lc in ("section_header", "heading"):
            pending_title = text or "(untitled)"
            cur = _new_section(pending_title)
            continue

        #Images
        if label_lc in ("picture", "image") or str(self_ref).startswith("#/pictures/"):
            if not in_excluded and cur:
                img = node.get("image") or {}
                uri = norm(img.get("uri")) if isinstance(img.get("uri"), str) else None
                path = _resolve_image_uri(uri) if uri else None
                if not path:
                    for k in ("file", "path", "filename", "file_name", "name"):
                        v = norm(node.get(k))
                        if v:
                            cand = (images_path / v) if not str(v).startswith(str(images_path)) else Path(v)
                            if cand.exists():
                                path = str(cand)
                                break
                if path:
                    cur["images"].append({"path": path})
            continue

        #Tables -> inject TSV into section text
        if label_lc in ("table",) or str(self_ref).startswith("#/tables/"):
            if not in_excluded:
                if cur is None:
                    cur = _new_section(pending_title or "Document")
                    if cur is None:
                        continue
                try:
                    tsv = _docling_table_to_tsv(node)  # None if structure missing
                    caption = norm(node.get("caption") or node.get("title") or "")
                    if tsv:
                        # Minimal guards so the LLM sees it as a block, not prose.
                        block = []
                        block.append(f"[TABLE BEGIN]{(' ' + caption) if caption else ''}")
                        block.append(tsv)
                        block.append("[TABLE END]")
                        cur["text"] += (("\n\n" if cur["text"] else "") + "\n".join(block))
                    else:
                        print(f"[error] Table parse failed (no structure) at {self_ref or '(unknown ref)'}; skipping.")
                except Exception as e:
                    print(f"[error] Exception while parsing table at {self_ref or '(unknown ref)'}: {e}")
            continue

        #Paragraph/caption
        if label_lc in {"text", "caption", "paragraph", "para", "body_text"} or (isinstance(text, str) and text.strip()):
            if in_excluded:
                continue
            if cur is None:
                # allow text before first header to go into a default section
                cur = _new_section(pending_title or "Document")
                # if excluded, _new_section returns None -> skip until a non-excluded header
                if cur is None:
                    continue
            if text.strip():
                cur["text"] += (("\n" if cur["text"] else "") + text.strip())
            continue

        #Ignore others

    return {"sections": out_sections}
#####------------------------------------------------------------------------------------------------------------------------#####



#####----- LLM Setup --------------------------------------------------------------------------------------------------------######
def load_model(modelID: str) -> Tuple[AutoProcessor, Llama4ForConditionalGeneration]:
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    #attn = os.environ.get("LLAMA_ATTN_IMPL", "flex_attention")
    processor = AutoProcessor.from_pretrained(modelID, trust_remote_code=True, use_fast=True, token=HF_TOKEN, cache_dir=CACHE_DIR)
    model = Llama4ForConditionalGeneration.from_pretrained(
        modelID,
        attn_implementation="sdpa",          
        device_map="auto",
        torch_dtype=torch.bfloat16,          
        trust_remote_code=True,
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
    )
    return processor, model

def prompt_model(processor, model, system: str, context_array: List[dict],
                 max_tokens: int, temperature: float) -> Tuple[str, Dict[str, Any]]:
    content = []
    loggable_content = []
    budget = VAR_maxContext

    for x in context_array:
        if x.get("type") == "text":
            txt = x.get("text", "")
            content.append({"type": "text", "text": txt})
            loggable_content.append({"type": "text", "text": txt})
            used = len(processor.tokenizer.encode(txt, add_special_tokens=False))
            budget = max(0, budget - used)
        elif x.get("type") == "image" and x.get("path"):
            try:
                with Image.open(x["path"]) as pil0:
                    pil = pil0.convert("RGB")
                    max_side = 1280
                    w, h = pil.size
                    if max(w, h) > max_side:
                        scale = max_side / float(max(w, h))
                        pil = pil.resize((int(w*scale), int(h*scale)))
                content.append({"type": "image", "image": pil})
            except Exception as e:
                print(f"[warn] failed to open image {x['path']}: {e}")
            loggable_content.append({"type": "image", "path": x["path"]})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user",   "content": content},
    ]
    loggable_messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user",   "content": loggable_content},
    ]

    rendered_prompt = processor.apply_chat_template(
        loggable_messages, add_generation_prompt=True, tokenize=False
    )

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    eos_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None) or getattr(model.config, "eos_token_id", None)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=bool(temperature and temperature > 0),
            temperature=float(temperature) if temperature else 0.0,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    out_text = processor.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)[0].strip()

    try:
        input_ids = inputs["input_ids"][0].to("cpu").tolist()
        token_count = len(input_ids)
    except Exception:
        token_count = None

    prompt_dump = {
        "messages": loggable_messages,              # structured; images as paths
        "rendered_chat_template": rendered_prompt,  # exact string form
        "token_count": token_count,
        "max_new_tokens": int(max_tokens),
        "temperature": float(temperature) if temperature else 0.0,
    }

    try:
        del inputs, outputs
        torch.cuda.empty_cache()
    except Exception:
        pass

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
        model=model,
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
    jpath = PATH_input / pdfname / f"{pdfname}.json"
    ipath = PATH_input / pdfname / f"{pdfname}_artifacts"

    parsed = parse_json(jpath, ipath, VAR_excludedSections)
    if debug:
        print(f"\nParsed json:\n\n{parsed}")

    #Build context (sections + images)
    context_base = []
    sections = parsed["sections"]
    imgs_used = 0
    images_total = sum(len(s.get("images", [])) for s in sections)
    warned = False

    for s in sections:
        context_base.append({"type": "text", "text": f"# {s['title']}\n{s['text']}"} )
        for im in s.get("images", []):
            if imgs_used >= VAR_maxImages:
                if not warned:
                    print(f"[info] Reached VAR_maxImages={VAR_maxImages}. "
                        f"Including only the first {imgs_used} of {images_total} images.")
                    warned = True
                break
            context_base.append({"type": "image", "path": im["path"]})
            imgs_used += 1

    if images_total > VAR_maxImages and not warned:
        print(f"[info] Reached VAR_maxImages={VAR_maxImages}. "
            f"Including only the first {imgs_used} of {images_total} images.")


    if processor is None or model is None:
        processor, model = load_model(VAR_model)

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

            answer, prompt_dump = prompt_model(processor, model, system_prompt, context, VAR_tokensPerAnswer, VAR_temperature)

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



processor, model = load_model(VAR_model)
pdfnames = ["2L7KQ4qbHi", "3lDxKQepvn", "eajZpoQkGK", "5WtovCb1ZE", "5ck9PIrTpH", "6T8czSBWce", "6ouZaBzeNO", "70lFRMBygi", "7JhGdZvW4T", "90z4EDqcmu", "tO58o0ahdg", "B6xUlbgP7j", "C9pndmSjg6", "b8eEutZlPb", "HhP9bgCugr", "J6qrIjTzoM", "Jy17uvzNe5", "KEeTRb8GLf", "KEqerdxOBG", "LIz0cBmHn5", "M1mL9tneGL", "MzmeLlDOkN", "60i0ksMAhd", "P7O1Vt1BdU", "Q47jVPzJ3G", "S5wIXxlvfw", "uNomADvF3s", "uBnM3EFovQ", "VdX9tL3VXH", "djnvlNHQR4", "XWBE90OYlH", "ReItdfwMcg", "bsXxNkhvm6", "cf7NTWv1iW", "gWqFbnKsqR", "gtCXzVeQxz", "h0ZfDIrj7T", "4X9RpKH4Ls", "j1jtyGdD4O", "iRYExPKnxm", "UwbX8KOZgK", "gWOANrFJ0t", "mmpVsDjVgn", "q44uq3tc2D", "qxRoo7ULCo", "GraOHDxFjl", "trj2Jq8riA", "D1gKsJagiq", "xriJVaTh4C", "0aaaM31hLB"]

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