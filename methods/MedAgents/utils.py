from .prompt_generator import *
from .data_utils import *



def fully_decode(qid, realqid, question, options, gold_answer, handler, args, dataobj,
                 num_qd=None, num_od=None, is_mca=False, role_mode="dynamic"):

    num_qd = num_qd if num_qd is not None else NUM_QD
    num_od = num_od if num_od is not None else NUM_OD

    question_domains, options_domains, question_analyses, option_analyses, syn_report, output = "", "", "", "", "", ""
    vote_history, revision_history, syn_repo_history = [], [], []
    current_num_agents=0

    is_vqa = not options or not options.strip()
    valid_letters = extract_option_letters(options) if not is_vqa else None

    if args.method == "base_direct":
        direct_prompt = get_direct_prompt(question, options)
        output = handler.get_output_multiagent(user_input=direct_prompt, temperature=0, max_tokens=50, system_role="")
        ans, output = cleansing_final_output_vqa(output) if is_vqa else cleansing_final_output(output, valid_letters=valid_letters)
    elif args.method == "base_cot":
        cot_prompt = get_cot_prompt(question, options)
        output = handler.get_output_multiagent(user_input=cot_prompt, temperature=0, max_tokens=300, system_role="")
        ans, output = cleansing_final_output_vqa(output) if is_vqa else cleansing_final_output(output, valid_letters=valid_letters)
    else:
        if role_mode == "fixed":
            question_domains = FIXED_QUESTION_DOMAINS[:num_qd]
            current_num_agents += 1
            print(f"\n[fixed mode] Using fixed question domains: {question_domains}")
        else:
            question_classifier, prompt_get_question_domain = get_question_domains_prompt(
                question, is_vqa=is_vqa, num_qd=num_qd, role_mode=role_mode)
            print(f"\nquestion_classifier: {question_classifier}")
            print(f"\nprompt_get_question_domain: {prompt_get_question_domain}")
            raw_question_domain = handler.get_output_multiagent(user_input=prompt_get_question_domain, temperature=0, max_tokens=1280, system_role=question_classifier)
            current_num_agents += 1
            print(f"\nraw_question_domain: {raw_question_domain}")
            if raw_question_domain == "ERROR.":
                raw_question_domain = "Medical Field: " + " | ".join(["General Medicine" for _ in range(num_qd)])
            question_domains = raw_question_domain.split(":")[-1].strip().split(" | ")

        vqa_analysis_max_tokens = 500 if is_vqa else 300
        tmp_question_analysis = []
        for _domain in question_domains:
            current_num_agents += 1
            question_analyzer, prompt_get_question_analysis = get_question_analysis_prompt(
                question, _domain, is_vqa=is_vqa, role_mode=role_mode)
            print(f"\nquestion_analyzer: {question_analyzer}")
            print(f"\nprompt_get_question_analysis: {prompt_get_question_analysis}")
            raw_question_analysis = handler.get_output_multiagent(user_input=prompt_get_question_analysis, temperature=0, max_tokens=vqa_analysis_max_tokens, system_role=question_analyzer)
            print(f"\nraw_question_analysis: {raw_question_analysis}")
            tmp_question_analysis.append(raw_question_analysis)
        question_analyses = cleansing_analysis(tmp_question_analysis, question_domains, 'question')

        if not is_vqa:
            if role_mode == "fixed":
                options_domains = FIXED_OPTION_DOMAINS[:num_od]
                current_num_agents += 1
                print(f"\n[fixed mode] Using fixed option domains: {options_domains}")
            else:
                options_classifier, prompt_get_options_domain = get_options_domains_prompt(
                    question, options, num_od=num_od, role_mode=role_mode)
                print(f"\noptions_classifier: {options_classifier}")
                print(f"\nprompt_get_options_domain: {prompt_get_options_domain}")
                raw_option_domain = handler.get_output_multiagent(user_input=prompt_get_options_domain, temperature=0, max_tokens=1280, system_role=options_classifier)
                current_num_agents+=1
                print(f"\nraw_option_domain: {raw_option_domain}")
                if raw_option_domain == "ERROR.":
                    raw_option_domain = "Medical Field: " + " | ".join(["General Medicine" for _ in range(num_od)])
                options_domains = raw_option_domain.split(":")[-1].strip().split(" | ")

            tmp_option_analysis = []
            for _domain in options_domains:
                current_num_agents += 1
                option_analyzer, prompt_get_options_analyses = get_options_analysis_prompt(
                    question, options, _domain, question_analyses, num_qd=num_qd, role_mode=role_mode)
                print(f"\noption_analyzer: {option_analyzer}")
                raw_option_analysis = handler.get_output_multiagent(user_input=prompt_get_options_analyses, temperature=0, max_tokens=300, system_role=option_analyzer)
                print(f"\nraw_option_analysis: {raw_option_analysis}")
                tmp_option_analysis.append(raw_option_analysis)
            option_analyses = cleansing_analysis(tmp_option_analysis, options_domains, 'option')
        else:
            print("[VQA mode] Skipping option domain/analysis (no options)")

        if args.method == "anal_only":
            answer_prompt = get_final_answer_prompt_analonly(question, options, question_analyses, option_analyses, num_qd=num_qd, num_od=num_od, is_mca=is_mca)
            output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=2500, system_role="")
            ans, output = cleansing_final_output_vqa(output) if is_vqa else (cleansing_final_output_mca(output, valid_letters=valid_letters) if is_mca else cleansing_final_output(output, valid_letters=valid_letters))
        else:
            q_analyses_text = transform_dict2text(question_analyses, "question", question)
            print(f"\nq_analyses_text: {q_analyses_text}")
            if is_vqa:
                synthesizer, prompt_get_synthesized_report = get_synthesized_report_prompt_vqa(
                    q_analyses_text, role_mode=role_mode)
            else:
                o_analyses_text = transform_dict2text(option_analyses, "options", options)
                print(f"\no_analyses_text: {o_analyses_text}")
                synthesizer, prompt_get_synthesized_report = get_synthesized_report_prompt(
                    q_analyses_text, o_analyses_text, role_mode=role_mode)
            print(f"\nsynthesizer: {synthesizer}")
            print(f"prompt_get_synthesized_report: {prompt_get_synthesized_report}")
            raw_synthesized_report = handler.get_output_multiagent(user_input=prompt_get_synthesized_report, temperature=0, max_tokens=2500, system_role=synthesizer)
            current_num_agents+=1
            print(f"\nraw_synthesized_report: {raw_synthesized_report}")
            syn_report = cleansing_syn_report(question, options, raw_synthesized_report)
            print(f"\nsyn_report: {syn_report}")

            if args.method == "syn_only":
                if is_vqa:
                    answer_prompt = get_final_answer_prompt_wsyn_vqa(syn_report)
                else:
                    answer_prompt = get_final_answer_prompt_wsyn(syn_report, options_text=options, is_mca=is_mca)
                output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=2500, system_role="")
                ans, output = cleansing_final_output_vqa(output) if is_vqa else (cleansing_final_output_mca(output, valid_letters=valid_letters) if is_mca else cleansing_final_output(output, valid_letters=valid_letters))
            elif args.method == "syn_verif":
                all_domains = question_domains if is_vqa else question_domains + options_domains

                syn_repo_history = [syn_report]

                hasno_flag = True
                num_try = 0

                while num_try < args.max_attempt_vote and hasno_flag:
                    domain_opinions = {}
                    revision_advice = {}
                    num_try += 1
                    hasno_flag = False
                    for domain in all_domains:
                        voter, cons_prompt = get_consensus_prompt(
                            domain, syn_report, is_vqa=is_vqa, role_mode=role_mode)
                        print(f"\nvoter: {voter}, cons_prompt: {cons_prompt}")
                        raw_domain_opi = handler.get_output_multiagent(user_input=cons_prompt, temperature=0, max_tokens=30, system_role=voter)
                        print(f"\nraw_domain_opi: {raw_domain_opi}")
                        domain_opinion = cleansing_voting(raw_domain_opi)
                        print(f"\ndomain_opinion: {domain_opinion}")
                        domain_opinions[domain] = domain_opinion
                        if domain_opinion == "no":
                            advice_prompt = get_consensus_opinion_prompt(
                                domain, syn_report, is_vqa=is_vqa, role_mode=role_mode)
                            print(f"\nadvice_prompt: {advice_prompt}")
                            advice_output = handler.get_output_multiagent(user_input=advice_prompt, temperature=0, max_tokens=500, system_role=voter)
                            print(f"\nadvice_output: {advice_output}")
                            revision_advice[domain] = advice_output
                            hasno_flag = True
                    if hasno_flag:
                        revision_prompt = get_revision_prompt(
                            syn_report, revision_advice, role_mode=role_mode)
                        print(f"\nrevision_prompt: {revision_prompt}")
                        revised_analysis = handler.get_output_multiagent(user_input=revision_prompt, temperature=0, max_tokens=2500, system_role="")
                        print(f"\nrevised_analysis: {revised_analysis}")
                        syn_report = cleansing_syn_report(question, options, revised_analysis)
                        print(f"\nsyn_report: {syn_report}")
                        revision_history.append(revision_advice)
                        syn_repo_history.append(syn_report)
                    vote_history.append(domain_opinions)
                
                print(f"\nsyn_report:{syn_report}")
                if is_vqa:
                    answer_prompt = get_final_answer_prompt_wsyn_vqa(syn_report)
                else:
                    answer_prompt = get_final_answer_prompt_wsyn(syn_report, options_text=options, is_mca=is_mca)
                print(f"\nanswer_prompt: {answer_prompt}")
                output = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, max_tokens=2500, system_role="")
                print(f"\noutput: {output}")
                ans, output = cleansing_final_output_vqa(output) if is_vqa else (cleansing_final_output_mca(output, valid_letters=valid_letters) if is_mca else cleansing_final_output(output, valid_letters=valid_letters))
                print(f"\nans: {ans}")
                print(f"\noutput: {output}")


    data_info = {
        'question': question,
        'options': options,
        'pred_answer': ans,
        'gold_answer': gold_answer,
        'question_domains': question_domains,
        'option_domains': options_domains,
        'question_analyses': question_analyses,
        'option_analyses': option_analyses,
        'syn_report': syn_report,
        'vote_history': vote_history,
        'revision_history': revision_history,
        'syn_repo_history': syn_repo_history,
        'raw_output': output,
        'current_num_agents': current_num_agents,
        'role_mode': role_mode,
    }
    
    return data_info

