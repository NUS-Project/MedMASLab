import os
from methods.Reconcile.model import *
from methods.Reconcile.utils import *




def Reconcile_test(question, root_path,model_info,img_path,batch_manager):
    # question="Q:"+question
    Reconcile = Reconcile_Model(root_path,model_info,batch_manager)
    Gemini_result = []
    tmp = {}
    # tmp['gold_answer'] = answer
    # convincing_Gemini = "Q:A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?, 'options':'A': 'Ampicillin','B': 'Ceftriaxone','C': 'Ciprofloxacin', 'D': 'Doxycycline', 'E':'Nitrofurantoin',\n A:{'reasoning':'The 23-year-old pregnant woman presents with symptoms consistent with a urinary tract infection (burning upon urination) at 22 weeks gestation without severe symptoms indicating pyelonephritis. Among the antibiotic options provided, Nitrofurantoin is effective for uncomplicated UTIs and is generally considered safe to use during pregnancy, unlike other options such as Ciprofloxacin and Doxycycline, which are contraindicated. Therefore, Nitrofurantoin is the best treatment choice.','answer':'E', 'confidence_level': 0.75}\n"
    result = Reconcile.Gemini_gen_ans(sample=question, convincing_samples=None, additional_instruc=None,
                                      intervene=False,img_path=img_path)
    tmp['prediction'] = result
    Gemini_result.append(tmp)

    gpt_result = []
    tmp = {}
    # tmp['gold_answer'] = answer
    # convincing_GPT = "Q:A mother brings her 3-week-old infant to the pediatrician's office because she is concerned about his feeding habits. He was born without complications and has not had any medical problems up until this time. However, for the past 4 days, he has been fussy, is regurgitating all of his feeds, and his vomit is yellow in color. On physical exam, the child's abdomen is minimally distended but no other abnormalities are appreciated. Which of the following embryologic errors could account for this presentation?'options':'A':'Abnormal migration of ventral pancreatic bud', 'B': 'Complete failure of proximal duodenum to recanalize', 'C': 'Error in neural crest cell migration', 'D': 'Abnormal hypertrophy of the pylorus', 'E': 'Failure of lateral body folds to move ventrally and fuse in the midline'\n A:{'reasoning':'The infant is presenting at 3 weeks of age with fussy behavior and 'yellow' (bilious) vomiting. Bilious vomiting in a neonate indicates an intestinal obstruction distal to the ampulla of Vater (second part of the duodenum). Option A (Annular pancreas) results from abnormal migration of the ventral pancreatic bud, which can cause duodenal obstruction and bilious vomiting. Option B (Duodenal atresia) results from failure of recanalization and presents with bilious vomiting, but usually within the first 24-48 hours of life and is often associated with Down syndrome. Option C (Hirschsprung disease) usually presents with failure to pass meconium and abdominal distension. Option D (Pyloric stenosis) presents with non-bilious projectile vomiting. Option E (Gastroschisis) is a defect seen at birth. Given the timeline and bilious nature of the vomit, an obstruction in the duodenum (like annular pancreas) is a classic embryologic cause for these symptoms.','answer':'A','confidence_level':0.90}"
    result = Reconcile.gpt_gen_ans(sample=question, convincing_samples=None, additional_instruc=None,
                                   intervene=False,img_path=img_path)

    tmp['prediction'] = result
    gpt_result.append(tmp)
    bard_result = []
    tmp = {}
    # tmp['gold_answer'] = answer
    # convincing_bard = "Q:A 3900-g (8.6-lb) male infant is delivered at 39 weeks' gestation via spontaneous vaginal delivery. Pregnancy and delivery were uncomplicated but a prenatal ultrasound at 20 weeks showed a defect in the pleuroperitoneal membrane. Further evaluation of this patient is most likely to show which of the following findings? options: 'A': 'Gastric fundus in the thorax', 'B': 'Pancreatic ring around the duodenum', 'C': 'Small and cystic kidneys', 'D': 'Hypertrophy of the gastric pylorus', 'E': 'Large bowel in the inguinal canal'\n A:{'reasoning': 'The infant's prenatal ultrasound at 20 weeks revealed a defect in the pleuroperitoneal membrane, which suggests the presence of a congenital diaphragmatic hernia. In such cases, the most common finding after delivery would be the displacement of abdominal contents into the thorax, which frequently includes the stomach. Thus, the expected finding on further evaluation would be a gastric fundus located in the thoracic cavity. The other options can be considered: Option B (pancreatic ring around the duodenum) relates to annular pancreas, which is unrelated to pleuroperitoneal defects; Option C (small and cystic kidneys) could relate to renal agenesis but does not correlate with the diaphragm defect; Option D (hypertrophy of the gastric pylorus) describes a different condition altogether (pyloric stenosis); Option E (large bowel in the inguinal canal) refers to an inguinal hernia, which is also distinct. Therefore, the correct answer is that the evaluation will show the gastric fundus in the thorax, indicating the effects of the diaphragmatic hernia.', 'final answer':'A', 'confidence_level': 0.95}"
    result = Reconcile.bard_gen_ans(sample=question, convincing_samples=None, additional_instruc=None,
                                    intervene=False,img_path=img_path)
    tmp['prediction'] = result
    bard_result.append(tmp)

    all_results = []
    for c, g, b in zip(Gemini_result, gpt_result, bard_result):
        tmp = {}
        # tmp['gold_answer'] = c['gold_answer']
        tmp['Gemini_output_0'] = c['prediction']
        tmp['gpt_output_0'] = g['prediction']
        tmp['bard_output_0'] = b['prediction']
        all_results.append(tmp)

    num_llm_calls=0
    prompt_tokens=0
    completion_tokens=0

    all_results,num_prompt,num_completion,num_call = clean_model_output(all_results, 0)
    # print(f"\nall_results: {all_results}")
    all_results = model_parse_output(all_results, 0)
    num_llm_calls+=num_call
    prompt_tokens+=num_prompt
    completion_tokens+=num_completion

    # print(f"\nall_results: {all_results}")
    # print(f"Initial Round Performance: {evaluate_model_all(all_results, 0)}")
    final_answer = all_results[0]["weighted_vote_0"]
    # print(f"\nfinal_answer: {final_answer}")
    max_key = max(final_answer, key=final_answer.get)
    # print(f"Max key{max_key}")
    rounds=1
    max_round=3
    for r in range(1, max_round):
        # print(f"----- Round {r} Discussion -----")
        all_results = Reconcile.Gemini_debate(test_samples=question,
                                              all_results=all_results,
                                              rounds=r,
                                              convincing_samples=None,img_path=img_path)

        all_results = Reconcile.gpt_debate(test_samples=question,
                                           all_results=all_results,
                                           rounds=r,
                                           convincing_samples=None,img_path=img_path)

        all_results = Reconcile.bard_debate(test_samples=question,
                                            all_results=all_results,
                                            rounds=r,
                                            convincing_samples=None,img_path=img_path)

        all_results,num_prompt,num_completion,num_call = clean_model_output(all_results, r)
        num_llm_calls+=num_call
        prompt_tokens+=num_prompt
        completion_tokens+=num_completion
        all_results = model_parse_output(all_results, r)
        # token_stats = Reconcile.get_token_stats()
        # print(f"\nall_results:{all_results}")
        # print(f"Round {r} Performance: {evaluate_model_all(all_results, r)}")
        final_answer = all_results[len(all_results) - 1][f"weighted_vote_{len(all_results) - 1}"]
        max_key = max(final_answer, key=final_answer.get)
        # print(f"Max key:{max_key}")
        # print(len(all_results[len(all_results) - 1][f"weighted_vote_{len(all_results) - 1}"]))
        if len(all_results[len(all_results) - 1][f"weighted_vote_{len(all_results) - 1}"]) == 1:
            rounds = r
            break


    token_stats = Reconcile.get_token_stats()
    token_stats["ReConcile"]["num_llm_calls"] +=num_llm_calls
    token_stats["ReConcile"]["prompt_tokens"] += prompt_tokens
    token_stats["ReConcile"]["completion_tokens"] += completion_tokens
    current_config = {"current_num_agents": 3, "round": rounds}

    return max_key, token_stats, current_config