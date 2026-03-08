import os
os.environ["GEMINI_API_KEY"] = "AIzaSyB9eXmCCPqBNTJ9UdixMpNKzkifIURafKg"
os.environ["OPEN_AI_API_KEY"] = "sk-YWOs2T3Qr5v1LgXMGzX4GWyJSn0Pqy19Ug8buK2cDIEVD1Wj"
os.environ["OPEN_AI_API_BASE"]="https://yinli.one/v1"
from model import *
from utils import *

question="Q:A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take? options:'A': 'Disclose the error to the patient but leave it out of the operative report', 'B': 'Disclose the error to the patient and put it in the operative report', 'C': 'Tell the attending that he cannot fail to disclose this mistake', 'D': 'Report the physician to the ethics committee', 'E': 'Refuse to dictate the operative report'"
answer= 'B'
# gold_explanation="Explanation :\nLet the height of the building be h. Initially, hewas at an angle of 450. tan 45 = h/distance between car and tower. h = distance between car and tower (since tan 45 = 1).\nNow, after 10 minutes, it travelled a certain distance, andangle changed to 600.\ntan 60 = h/x x = h/√3\nSo, in 10 minutes, it has travelled a distance of h – x = h - h/√3.\n10 minutes = h *( 1 – 1√3)\nh can be travelled in 10 / (1 – 1√3).\nTo travel a distance of x, which is h/√3, it takes :\nh = 10 / (1 – 1/√3)\nh / √3 = 10/ √3 * (1 – 1/√3). Multiply numerator and denominator by 1 + √3 ( conjugate of 1 - √3). We get, x = h/√3 = 10 (1 + √3) / 2 = 5* (1 + √3)\nSo, it takes 5(1 + √3) minutes to reach the base of the tower.\nAnswer : A"
Reconcile = Reconcile_Model()
Gemini_result = []
tmp = {}
tmp['gold_answer'] = answer
convincing_Gemini="Q:A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?, 'options':'A': 'Ampicillin','B': 'Ceftriaxone','C': 'Ciprofloxacin', 'D': 'Doxycycline', 'E':'Nitrofurantoin',\n A:{'reasoning':'The 23-year-old pregnant woman presents with symptoms consistent with a urinary tract infection (burning upon urination) at 22 weeks gestation without severe symptoms indicating pyelonephritis. Among the antibiotic options provided, Nitrofurantoin is effective for uncomplicated UTIs and is generally considered safe to use during pregnancy, unlike other options such as Ciprofloxacin and Doxycycline, which are contraindicated. Therefore, Nitrofurantoin is the best treatment choice.','answer':'E', 'confidence_level': 0.75}\n"
result =Reconcile.Gemini_gen_ans(sample=question,convincing_samples=convincing_Gemini,additional_instruc=None,intervene=False)
tmp['prediction'] = result
Gemini_result.append(tmp)

gpt_result = []
tmp = {}
tmp['gold_answer'] = answer
convincing_GPT = "Q:A mother brings her 3-week-old infant to the pediatrician's office because she is concerned about his feeding habits. He was born without complications and has not had any medical problems up until this time. However, for the past 4 days, he has been fussy, is regurgitating all of his feeds, and his vomit is yellow in color. On physical exam, the child's abdomen is minimally distended but no other abnormalities are appreciated. Which of the following embryologic errors could account for this presentation?'options':'A':'Abnormal migration of ventral pancreatic bud', 'B': 'Complete failure of proximal duodenum to recanalize', 'C': 'Error in neural crest cell migration', 'D': 'Abnormal hypertrophy of the pylorus', 'E': 'Failure of lateral body folds to move ventrally and fuse in the midline'\n A:{'reasoning':'The infant is presenting at 3 weeks of age with fussy behavior and 'yellow' (bilious) vomiting. Bilious vomiting in a neonate indicates an intestinal obstruction distal to the ampulla of Vater (second part of the duodenum). Option A (Annular pancreas) results from abnormal migration of the ventral pancreatic bud, which can cause duodenal obstruction and bilious vomiting. Option B (Duodenal atresia) results from failure of recanalization and presents with bilious vomiting, but usually within the first 24-48 hours of life and is often associated with Down syndrome. Option C (Hirschsprung disease) usually presents with failure to pass meconium and abdominal distension. Option D (Pyloric stenosis) presents with non-bilious projectile vomiting. Option E (Gastroschisis) is a defect seen at birth. Given the timeline and bilious nature of the vomit, an obstruction in the duodenum (like annular pancreas) is a classic embryologic cause for these symptoms.','answer':'A','confidence_level':0.90}"
result = Reconcile.gpt_gen_ans(sample=question,convincing_samples=convincing_GPT,additional_instruc=None,intervene=False)

tmp['prediction'] = result
gpt_result.append(tmp)
bard_result = []
tmp = {}
tmp['gold_answer'] = answer
convincing_bard ="Q:A 3900-g (8.6-lb) male infant is delivered at 39 weeks' gestation via spontaneous vaginal delivery. Pregnancy and delivery were uncomplicated but a prenatal ultrasound at 20 weeks showed a defect in the pleuroperitoneal membrane. Further evaluation of this patient is most likely to show which of the following findings? options: 'A': 'Gastric fundus in the thorax', 'B': 'Pancreatic ring around the duodenum', 'C': 'Small and cystic kidneys', 'D': 'Hypertrophy of the gastric pylorus', 'E': 'Large bowel in the inguinal canal'\n A:{'reasoning': 'The infant's prenatal ultrasound at 20 weeks revealed a defect in the pleuroperitoneal membrane, which suggests the presence of a congenital diaphragmatic hernia. In such cases, the most common finding after delivery would be the displacement of abdominal contents into the thorax, which frequently includes the stomach. Thus, the expected finding on further evaluation would be a gastric fundus located in the thoracic cavity. The other options can be considered: Option B (pancreatic ring around the duodenum) relates to annular pancreas, which is unrelated to pleuroperitoneal defects; Option C (small and cystic kidneys) could relate to renal agenesis but does not correlate with the diaphragm defect; Option D (hypertrophy of the gastric pylorus) describes a different condition altogether (pyloric stenosis); Option E (large bowel in the inguinal canal) refers to an inguinal hernia, which is also distinct. Therefore, the correct answer is that the evaluation will show the gastric fundus in the thorax, indicating the effects of the diaphragmatic hernia.', 'final answer':'A', 'confidence_level': 0.95}"
result = Reconcile.bard_gen_ans(sample=question,convincing_samples=convincing_bard,additional_instruc=None,intervene=False)
tmp['prediction'] = result
bard_result.append(tmp)

all_results = []
for c, g, b in zip(Gemini_result, gpt_result, bard_result):
    tmp = {}
    tmp['gold_answer'] = c['gold_answer']
    tmp['Gemini_output_0'] = c['prediction']
    tmp['gpt_output_0'] = g['prediction']
    tmp['bard_output_0'] = b['prediction']
    all_results.append(tmp)

all_results = clean_model_output(all_results, 0)
print(f"\nall_results: {all_results}")
all_results = model_parse_output(all_results, 0)
print(f"\nall_results: {all_results}")
# print(f"Initial Round Performance: {evaluate_model_all(all_results, 0)}")
final_answer = all_results[0]["weighted_vote_0"]
print(f"\nfinal_answer: {final_answer}")
max_key = max(final_answer, key=final_answer.get)
print(f"Max key{max_key}")

for r in range(1, 3):
    print(f"----- Round {r} Discussion -----")
    all_results = Reconcile.Gemini_debate(test_samples=question,
                                       all_results=all_results,
                                       rounds=r,
                                       convincing_samples=convincing_Gemini)

    all_results = Reconcile.gpt_debate(test_samples=question,
                             all_results=all_results,
                             rounds=r,
                             convincing_samples=convincing_GPT)

    all_results = Reconcile.bard_debate(test_samples=question,
                              all_results=all_results,
                              rounds=r,
                              convincing_samples=convincing_bard)

    all_results = clean_model_output(all_results, r)
    all_results = model_parse_output(all_results, r)
    token_stats=Reconcile.get_token_stats()
    print(token_stats["ReConcile"]["num_llm_calls"])
    print(token_stats["ReConcile"]["prompt_tokens"])
    print(token_stats["ReConcile"]["completion_tokens"])
    print(all_results)
    print(f"Round {r} Performance: {evaluate_model_all(all_results, r)}")
    final_answer = all_results[len(all_results)-1][f"weighted_vote_{len(all_results)-1}"]
    max_key = max(final_answer, key=final_answer.get)
    print(f"Max key{max_key}")
#
# with open(f'{args.dataset}_round_{args.round}.pkl', 'wb') as f:
#     pickle.dump(all_results, f)