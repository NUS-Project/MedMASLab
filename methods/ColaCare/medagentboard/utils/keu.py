from typing import List, Dict, Any

class KEU:
    def __init__(self, keu_id: str, content: str, source_agent: str, round_introduced: int):
        self.keu_id: str = keu_id
        self.content: str = content
        self.source_agent: str = source_agent
        self.round_introduced: int = round_introduced
        
        self.is_key: bool = False 
        
        self.cited_by: List[Dict[str, Any]] = []  # e.g., {'agent_id': 'meta', 'round': 1, 'action': 'synthesis'}
        
        self.rebuttals: List[Dict[str, Any]] = [] # e.g., {'agent_id': 'doctor_2', 'round': 1, 'reason': '...'}
        
        self.present_in_synthesis: Dict[int, bool] = {} # {round_num: True/False}
        self.present_in_final_decision: bool = False

    def to_dict(self):
        return self.__dict__