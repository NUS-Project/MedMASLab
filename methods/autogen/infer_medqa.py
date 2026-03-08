from __future__ import annotations
from methods.maslab_runtime_config import build_general_config
from .autogen_main import AutoGen_Main


def autogen_infer_medqa(question, root_path, model_info, img_paths=None,batch_manager=None):
    # question, root_path, model_name,
    #                                    img_paths=img_path,batch_manager=batch_mgr
    mas = AutoGen_Main(model_name="autogen",batch_manager=batch_manager)

    result, current_config = mas.inference({"query": question, "img_paths": img_paths})
    response = result.get("response", "")
    # print(f"autogen_response:{response}")
    token_stats = mas.get_token_stats()
    return str(response).strip(), token_stats, current_config
