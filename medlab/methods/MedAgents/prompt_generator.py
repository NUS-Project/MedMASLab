
DEFAULT_NUM_QD = 5
DEFAULT_NUM_OD = 2

NUM_QD = DEFAULT_NUM_QD
NUM_OD = DEFAULT_NUM_OD

FIXED_QUESTION_DOMAINS = [
    "Internal Medicine", "Surgery", "Radiology", "Cardiology", "Neurology",
]
FIXED_OPTION_DOMAINS = [
    "Pharmacology", "Pathology",
]

_NUM_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def _num_word(n):
    return _NUM_WORDS.get(n, str(n))


def get_question_domains_prompt(question, is_vqa=False, num_qd=None, role_mode="dynamic"):
    n = num_qd if num_qd is not None else NUM_QD
    question_domain_format = "Medical Field: " + " | ".join(["Field" + str(i) for i in range(n)])
    nw = _num_word(n)
    if role_mode == "norole":
        question_classifier = "You are a helpful assistant who categorizes questions into different analytical perspectives."
        prompt_get_question_domain = (
            f"You need to complete the following steps:"
            f"1. Carefully read the question: '''{question}'''. \n"
            f"2. Classify the question into {nw} different analytical perspectives. \n"
            f"3. You should output in exactly the same format as '''{question_domain_format}'''."
        )
    elif is_vqa:
        question_classifier = "You are a medical expert who specializes in analyzing medical images and categorizing clinical questions into specific areas of medicine."
        prompt_get_question_domain = f"You need to complete the following steps:" \
                f"1. A medical image is provided along with the following clinical question: '''{question}'''. \n" \
                f"2. Based on the medical image and the question, classify it into {nw} different subfields of medicine most relevant to answering the question. \n" \
                f"3. You should output in exactly the same format as '''{question_domain_format}'''."
    else:
        question_classifier = "You are a medical expert who specializes in categorizing a specific medical scenario into specific areas of medicine."
        prompt_get_question_domain = f"You need to complete the following steps:" \
                f"1. Carefully read the medical scenario presented in the question: '''{question}'''. \n" \
                f"2. Based on the medical scenario in it, classify the question into {nw} different subfields of medicine. \n" \
                f"3. You should output in exactly the same format as '''{question_domain_format}'''."
    return question_classifier, prompt_get_question_domain



def get_question_analysis_prompt(question, question_domain, is_vqa=False, role_mode="dynamic"):
    if role_mode == "norole":
        question_analyzer = (
            f"You are a helpful assistant. "
            f"From your general knowledge, you will analyze questions and provide your perspective."
        )
        if is_vqa:
            prompt_get_question_analysis = (
                f"An image is provided along with the following question: '''{question}'''.\n"
                f"Please carefully examine the image first. Describe the key findings you observe. "
                f"Then provide a focused analysis of the findings that are most relevant to answering the question."
            )
        else:
            prompt_get_question_analysis = (
                f"Please meticulously examine the scenario outlined in this question: '''{question}'''."
                f"Interpret the situation being depicted. "
                f"Subsequently, identify and highlight the aspects of the issue that you find most noteworthy."
            )
    elif is_vqa:
        question_analyzer = f"You are a medical expert in the domain of {question_domain}. " \
            f"From your area of specialization, you will carefully analyze the provided medical image to answer clinical questions."
        prompt_get_question_analysis = f"A medical image is provided along with the following clinical question: '''{question}'''.\n" \
                            f"Please carefully examine the medical image first. Describe the key findings you observe in the image. " \
                            f"Then, drawing upon your medical expertise in {question_domain}, provide a focused analysis " \
                            f"of the findings that are most relevant to answering the question."
    else:
        question_analyzer = f"You are a medical expert in the domain of {question_domain}. " \
            f"From your area of specialization, you will scrutinize and diagnose the symptoms presented by patients in specific medical scenarios."
        prompt_get_question_analysis = f"Please meticulously examine the medical scenario outlined in this question: '''{question}'''." \
                            f"Drawing upon your medical expertise, interpret the condition being depicted. " \
                            f"Subsequently, identify and highlight the aspects of the issue that you find most alarming or noteworthy."

    return question_analyzer, prompt_get_question_analysis

def get_options_domains_prompt(question, options, num_od=None, role_mode="dynamic"):
    n = num_od if num_od is not None else NUM_OD
    options_domain_format = "Medical Field: " + " | ".join(["Field" + str(i) for i in range(n)])
    nw = _num_word(n)
    if role_mode == "norole":
        options_classifier = (
            f"You are a helpful assistant. You possess the ability to discern the {nw} most relevant "
            f"analytical perspectives needed to address a multiple-choice question."
        )
        prompt_get_options_domain = (
            f"You need to complete the following steps:"
            f"1. Carefully read the scenario presented in the question: '''{question}'''."
            f"2. The available options are: '''{options}'''. Strive to understand the connections between the question and the options."
            f"3. Categorize the options into {nw} distinct analytical perspectives. "
            f"You should output in exactly the same format as '''{options_domain_format}'''"
        )
    else:
        options_classifier = f"As a medical expert, you possess the ability to discern the {nw} most relevant fields of expertise needed to address a multiple-choice question encapsulating a specific medical context."
        prompt_get_options_domain = f"You need to complete the following steps:" \
                    f"1. Carefully read the medical scenario presented in the question: '''{question}'''." \
                    f"2. The available options are: '''{options}'''. Strive to understand the fundamental connections between the question and the options." \
                    f"3. Your core aim should be to categorize the options into {nw} distinct subfields of medicine. " \
                    f"You should output in exactly the same format as '''{options_domain_format}'''"
    return options_classifier, prompt_get_options_domain


def get_options_analysis_prompt(question, options, op_domain, question_analysis, num_qd=None, role_mode="dynamic"):
    n_qd = num_qd if num_qd is not None else NUM_QD
    if role_mode == "norole":
        option_analyzer = (
            f"You are a helpful assistant. "
            f"You are adept at comprehending the nexus between questions and choices in multiple-choice exams and determining their validity. "
            f"Your task is to analyze individual options and evaluate their relevancy and correctness."
        )
    else:
        option_analyzer = f"You are a medical expert specialized in the {op_domain} domain. " \
                    f"You are adept at comprehending the nexus between questions and choices in multiple-choice exams and determining their validity. " \
                    f"Your task, in particular, is to analyze individual options with your expert medical knowledge and evaluate their relevancy and correctness."

    prompt_get_options_analyses = f"Regarding the question: '''{question}''', we procured the analysis of {_num_word(n_qd)} experts from diverse domains. \n"
    for _domain, _analysis in question_analysis.items():
        prompt_get_options_analyses += f"The evaluation from the {_domain} expert suggests: {_analysis} \n"
        prompt_get_options_analyses += f"The following are the options available: '''{options}'''." \
                    f"Reviewing the question's analysis from the expert team, you're required to fathom the connection between the options and the question from the perspective of your respective domain, " \
                    f"and scrutinize each option individually to assess whether it is plausible or should be eliminated based on reason and logic. "\
                    f"Pay close attention to discerning the disparities among the different options and rationalize their existence. " \
                    f"A handful of these options might seem right on the first glance but could potentially be misleading in reality."
    return option_analyzer, prompt_get_options_analyses


def extract_option_letters(options_text):
    """Extract valid option letters from options text like '(A) ... (B) ... (C) ...'."""
    import re
    letters = re.findall(r'\(([A-Z])\)', str(options_text))
    return letters if letters else ['A', 'B', 'C', 'D']


def format_option_letters(options_text):
    """Return a display string of valid option letters, e.g. 'A, B, C, or D'."""
    letters = extract_option_letters(options_text)
    if len(letters) <= 1:
        return ', '.join(letters)
    return ', '.join(letters[:-1]) + ', or ' + letters[-1]


def get_final_answer_prompt_analonly(question, options, question_analyses, option_analyses, num_qd=None, num_od=None, is_mca=False):
    n_qd = num_qd if num_qd is not None else NUM_QD
    n_od = num_od if num_od is not None else NUM_OD
    valid_letters = format_option_letters(options)
    prompt = f"Question: {question} \nOptions: {options} \n" \
        f"Answer: Let's work this out in a step by step way to be sure we have the right answer. \n" \
        f"Step 1: Decode the question properly. We have a team of experts who have done a detailed analysis of this question. " \
        f"The team includes {_num_word(n_qd)} experts from different medical domains related to the problem. \n"
    
    for _domain, _analysis in question_analyses.items():
        prompt += f"Insight from an expert in {_domain} suggests, {_analysis} \n"
    
    prompt += f"Step 2: Evaluate each presented option individually, based on both the specifics of the patient's scenario as well as your medical knowledge. " \
            f"Pay close attention to discerning the disparities among the different options. " \
            f"A handful of these options might seem right on the first glance but could potentially be misleading in reality. " \
            f"We have detailed analyses from experts across {_num_word(n_od)} domains. \n"
    
    for _domain, _analysis in option_analyses.items():
        prompt += f"Assessment from an expert in {_domain} suggests, {_analysis} \n"

    if is_mca:
        prompt += f"Step 3: Based on the understanding gathered from the above steps, select ALL correct options to answer the question. \n" \
            f"Points to note: \n" \
            f"1. The analyses provided should guide you towards the correct response. \n" \
            f"2. This is a MULTI-SELECT question. You MUST select ALL options that apply. \n" \
            f"3. Please respond with ALL selected option letters separated by commas, using the following format: '''Option: A, B, C'''. " \
            f"Remember, list ALL correct letters separated by commas."
    else:
        prompt += f"Step 3: Based on the understanding gathered from the above steps, select the optimal choice to answer the question. \n" \
            f"Points to note: \n" \
            f"1. The analyses provided should guide you towards the correct response. \n" \
            f"2. Any option containing incorrect information inherently cannot be the correct choice. \n" \
            f"3. Please respond only with the selected option's letter, like {valid_letters}, using the following format: '''Option: [Selected Option's Letter]'''. " \
            f"Remember, it's the letter we need, not the full content of the option."

    return prompt

def get_final_answer_prompt_wsyn(syn_report, options_text="", is_mca=False):
    valid_letters = format_option_letters(options_text) if options_text else "A, B, C, or D"

    if is_mca:
        prompt = f"Here is a synthesized report: {syn_report} \n" \
            f"Based on the above report, select ALL correct options to answer the question. \n" \
            f"Points to note: \n" \
            f"1. The analyses provided should guide you towards the correct response. \n" \
            f"2. This is a MULTI-SELECT question. You MUST select ALL options that apply. \n" \
            f"3. Please respond with ALL selected option letters separated by commas, using the following format: '''Option: A, B, C'''. " \
            f"Remember, list ALL correct letters separated by commas."
    else:
        prompt = f"Here is a synthesized report: {syn_report} \n" \
            f"Based on the above report, select the optimal choice to answer the question. \n" \
            f"Points to note: \n" \
            f"1. The analyses provided should guide you towards the correct response. \n" \
            f"2. Any option containing incorrect information inherently cannot be the correct choice. \n" \
            f"3. Please respond only with the selected option's letter, like {valid_letters}, using the following format: '''Option: [Selected Option's Letter]'''. " \
            f"Remember, it's the letter we need, not the full content of the option."
    return prompt


def get_final_answer_prompt_wsyn_vqa(syn_report):
    """VQA version: asks for a direct text answer instead of an option letter."""
    prompt = f"Here is a synthesized report from multiple medical experts: {syn_report} \n" \
        f"A medical image is also provided for your reference. " \
        f"Based on BOTH the expert report AND the medical image, provide a direct and concise answer to the question. \n" \
        f"Points to note: \n" \
        f"1. Carefully look at the medical image to verify findings before answering. \n" \
        f"2. If the question asks yes/no, answer ONLY with 'Yes' or 'No'. \n" \
        f"3. If the question asks about a specific finding, location, or description, answer with a brief specific phrase. \n" \
        f"4. Please respond using the following format: '''Answer: [Your concise answer]'''. " \
        f"Keep your answer as brief as possible (e.g., 'Yes', 'No', 'Left lower lobe', 'Pulmonary nodules', 'CT scan')."
    return prompt


def get_direct_prompt(question, options):
    valid_letters = format_option_letters(options)
    prompt = f"Question: {question} \n" \
        f"Options: {options} \n" \
        f"Please respond only with the selected option's letter, like {valid_letters}, using the following format: '''Option: [Selected Option's Letter]'''."
    return prompt

def get_cot_prompt(question, options):
    valid_letters = format_option_letters(options)
    cot_format = f"Thought: [the step-by-step thoughts] \n" \
                f"Answer: [Selected Option's Letter (like {valid_letters})] \n"
    prompt = f"Question: {question} \n" \
        f"Options: {options} \n" \
        f"Answer: Let's work this out in a step by step way to be sure we have the right answer. " \
        f"You should output in exactly the same format as '''{cot_format}'''"
    return prompt


def get_synthesized_report_prompt(question_analyses, option_analyses, role_mode="dynamic"):
    if role_mode == "norole":
        synthesizer = "You are a helpful assistant who excels at summarizing and synthesizing reports from multiple perspectives."
    else:
        synthesizer = "You are a medical decision maker who excels at summarizing and synthesizing based on multiple experts from various domain experts."

    syn_report_format = f"Key Knowledge: [extracted key knowledge] \n" \
                f"Total Analysis: [synthesized analysis] \n"
    prompt = f"Here are some reports from different {'perspectives' if role_mode == 'norole' else 'medical domain experts'}.\n "
    prompt += f"You need to complete the following steps:" \
                f"1. Take careful and comprehensive consideration of the following reports." \
                f"2. Extract key knowledge from the following reports. " \
                f"3. Derive the comprehensive and summarized analysis based on the knowledge." \
                f"4. Your ultimate goal is to derive a refined and synthesized report based on the following reports." \
                f"You should output in exactly the same format as '''{syn_report_format}'''"
    prompt += question_analyses
    prompt += option_analyses
    
    return synthesizer, prompt


def get_synthesized_report_prompt_vqa(question_analyses, role_mode="dynamic"):
    """VQA version: synthesize report from question analyses only (no options)."""
    if role_mode == "norole":
        synthesizer = "You are a helpful assistant. An image has been provided. " \
                      "You excel at synthesizing image analyses from multiple perspectives to form an accurate assessment."
    else:
        synthesizer = "You are a medical decision maker. A medical image has been provided. " \
                      "You excel at synthesizing image analyses from multiple domain experts to form an accurate medical assessment."

    syn_report_format = f"Key Knowledge: [key findings observed in the image] \n" \
                f"Total Analysis: [synthesized analysis that directly answers the question based on image findings] \n"
    if role_mode == "norole":
        prompt = f"An image has been provided along with analyses from different perspectives.\n"
    else:
        prompt = f"A medical image has been provided along with analyses from different medical domain experts.\n"
    prompt += f"You need to complete the following steps:" \
                f"1. Carefully examine the image provided. " \
                f"2. Take careful consideration of the following reports about the image. " \
                f"3. Extract key findings from both the image and the reports. " \
                f"4. Derive a focused and accurate analysis that directly addresses the question. " \
                f"Prioritize what you can directly observe in the image over any conflicting opinions." \
                f"You should output in exactly the same format as '''{syn_report_format}'''"
    prompt += question_analyses
    
    return synthesizer, prompt


def get_consensus_prompt(domain, syn_report, is_vqa=False, role_mode="dynamic"):
    if role_mode == "norole":
        voter = "You are a helpful assistant."
    else:
        voter = f"You are a medical expert specialized in the {domain} domain."
    if is_vqa:
        if role_mode == "norole":
            cons_prompt = f"Here is a report based on an image: {syn_report} \n"\
                f"Please examine the image and the report. " \
                f"Decide whether the report accurately describes the findings visible in the image. " \
                f"Please respond only with: [YES or NO]."
        else:
            cons_prompt = f"Here is a medical report based on a medical image: {syn_report} \n"\
                f"As a medical expert specialized in {domain}, please examine the medical image and the report. " \
                f"Decide whether the report accurately describes the findings visible in the image. " \
                f"Please respond only with: [YES or NO]."
    else:
        if role_mode == "norole":
            cons_prompt = f"Here is a report: {syn_report} \n"\
                f"Please carefully read the report and decide whether your opinions are consistent with this report." \
                f"Please respond only with: [YES or NO]."
        else:
            cons_prompt = f"Here is a medical report: {syn_report} \n"\
                f"As a medical expert specialized in {domain}, please carefully read the report and decide whether your opinions are consistent with this report." \
                f"Please respond only with: [YES or NO]."
    return voter, cons_prompt


def get_consensus_opinion_prompt(domain, syn_report, is_vqa=False, role_mode="dynamic"):
    if is_vqa:
        if role_mode == "norole":
            opinion_prompt = f"Here is a report based on an image: {syn_report} \n"\
                f"Please examine the image carefully and propose revisions to make the report more accurately reflect the image findings." \
                f"You should output in exactly the same format as '''Revisions: [proposed revision advice] '''"
        else:
            opinion_prompt = f"Here is a medical report based on a medical image: {syn_report} \n"\
                f"As a medical expert specialized in {domain}, please examine the image carefully and propose revisions to make the report more accurately reflect the image findings." \
                f"You should output in exactly the same format as '''Revisions: [proposed revision advice] '''"
    else:
        if role_mode == "norole":
            opinion_prompt = f"Here is a report: {syn_report} \n"\
                f"Please propose revisions to this report." \
                f"You should output in exactly the same format as '''Revisions: [proposed revision advice] '''"
        else:
            opinion_prompt = f"Here is a medical report: {syn_report} \n"\
                f"As a medical expert specialized in {domain}, please make full use of your expertise to propose revisions to this report." \
                f"You should output in exactly the same format as '''Revisions: [proposed revision advice] '''"
    return opinion_prompt


def get_revision_prompt(syn_report, revision_advice, role_mode="dynamic"):
    revision_prompt = f"Here is the original report: {syn_report}\n\n"
    for domain, advice in revision_advice.items():
        if role_mode == "norole":
            revision_prompt += f"Here is advice from an analyst: {advice}.\n"
        else:
            revision_prompt += f"Here is advice from a medical expert specialized in {domain}: {advice}.\n"
    revision_prompt += f"Based on the above advice, output the revised analysis in exactly the same format as '''Total Analysis: [revised analysis] '''"
    return revision_prompt