"""
custom_mas_ui.py (增强版 - 支持删除连接)
================
Interactive drag-and-drop Multi-Agent System Designer for MedMASLab.
Features:
  - ⚠️  MANDATORY Input and Output nodes (auto-generated)
  - Drag Agent nodes from palette
  - Connect nodes with arrows pointing from left edge
  - Adjust node position using X/Y sliders
  - DELETE nodes via dropdown + button (left panel)
  - DELETE edges (connections) via dropdown + button (left panel)
  - Visualize the system topology
  - Generate executable code
"""

import json
import html
import math
import gradio as gr
from typing import Dict, List, Tuple, Optional
from pathlib import Path  # ✨ 新增
from datetime import datetime  # ✨ 新增
import base64  # ✨ 新增
from openai import OpenAI  # ✨ 新增

from io import BytesIO  # ✨ 新增
try:
    import cairosvg
except ImportError:
    cairosvg = None
# ═══════════════════════════════════════════════════════════════════
# Data Structures for Flow Design
# ═══════════════════════════════════════════════════════════════════
# 在文件顶部添加全局变量存储
_api_config = {
    "base_url": "",
    "api_key": "",
    "model_name": ""
}

def set_api_config(base_url: str, api_key: str, model_name: str):
    """保存 API 配置到全局变量"""
    global _api_config
    _api_config = {
        "base_url": base_url,
        "api_key": api_key,
        "model_name": model_name
    }

def get_api_config_from_web():
    """获取已保存的 API 配置"""
    return _api_config.copy()

class MASDesigner:
    """Manages the Multi-Agent System design state."""

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}  # id -> {type, label, x, y}
        self.edges: List[Tuple[str, str]] = []  # List of (from_id, to_id)
        self.next_node_id = 0
        self.selected_node_id = None  # Track selected node for position adjustment

        # 初始化时创建强制的 Input 和 Output 节点
        self._init_mandatory_nodes()

    def _init_mandatory_nodes(self):
        """Initialize mandatory Input and Output nodes."""
        # Input node on the left
        self.nodes["input"] = {
            "type": "Input",
            "label": "Input",
            "x": 50.0,
            "y": 250.0,
            "mandatory": True,
        }

        # Output node on the right
        self.nodes["output"] = {
            "type": "Output",
            "label": "Output",
            "x": 650.0,
            "y": 250.0,
            "mandatory": True,
        }

        self.next_node_id = 0

    def add_node(self, agent_type: str, x: float = 0, y: float = 0, prompt: str = "") -> str:
        """Add a new agent node and return its ID."""
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        self.nodes[node_id] = {
            "type": agent_type,
            "label": agent_type,
            "x": x,
            "y": y,
            "mandatory": False,
            "prompt": prompt,  # ✨ 新增
        }
        return node_id

    def select_node(self, node_id: str):
        """Select a node for position adjustment."""
        if node_id in self.nodes:
            # 不允许选择强制节点进行位置调整
            if self.nodes[node_id].get("mandatory", False):
                self.selected_node_id = None
            else:
                self.selected_node_id = node_id
        else:
            self.selected_node_id = None

    def get_selected_node_pos(self) -> Tuple[float, float]:
        """Get current selected node's position."""
        if self.selected_node_id and self.selected_node_id in self.nodes:
            node = self.nodes[self.selected_node_id]
            return float(node["x"]), float(node["y"])
        return 0.0, 0.0

    def move_node(self, node_id: str, x: float, y: float):
        """Update node position."""
        if node_id in self.nodes:
            # 不允许移动强制节点
            if self.nodes[node_id].get("mandatory", False):
                return

            # Clamp to canvas bounds
            x = max(0, min(x, 700))
            y = max(0, min(y, 500))
            self.nodes[node_id]["x"] = float(x)
            self.nodes[node_id]["y"] = float(y)

    def get_node_prompt(self, node_id: str) -> str:
        """Get the prompt of a specific node."""
        if node_id in self.nodes:
            return self.nodes[node_id].get("prompt", "")
        return ""

    def update_node_prompt(self, node_id: str, new_prompt: str) -> bool:
        """Update the prompt of a specific node."""
        if node_id in self.nodes:
            self.nodes[node_id]["prompt"] = new_prompt
            return True
        return False

    def add_edge(self, from_id: str, to_id: str):
        """Add a connection between two nodes."""
        if from_id in self.nodes and to_id in self.nodes:
            if (from_id, to_id) not in self.edges:
                self.edges.append((from_id, to_id))

    def remove_edge(self, from_id: str, to_id: str) -> bool:
        """Remove a connection. Returns True if removed, False if not found."""
        if (from_id, to_id) in self.edges:
            self.edges.remove((from_id, to_id))
            return True
        return False

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its connections (but not mandatory ones)."""
        if node_id not in self.nodes:
            return False

        # 不允许删除强制节点
        if self.nodes[node_id].get("mandatory", False):
            return False

        del self.nodes[node_id]
        self.edges = [(f, t) for f, t in self.edges if f != node_id and t != node_id]
        if self.selected_node_id == node_id:
            self.selected_node_id = None
        return True

    def clear(self):
        """Clear all nodes and edges, but keep mandatory Input/Output."""
        self.nodes.clear()
        self.edges.clear()
        self.next_node_id = 0
        self.selected_node_id = None
        self._init_mandatory_nodes()

    def to_dict(self) -> Dict:
        """Export design as dictionary."""
        return {
            "nodes": self.nodes.copy(),
            "edges": self.edges.copy(),
        }

    def from_dict(self, data: Dict):
        """Import design from dictionary."""
        self.clear()  # Clear and reinit mandatory nodes
        self.nodes.update(data.get("nodes", {}))
        self.edges = data.get("edges", [])
        # Preserve mandatory nodes
        if "input" not in self.nodes:
            self.nodes["input"] = {
                "type": "Input",
                "label": "Input",
                "x": 50.0,
                "y": 250.0,
                "mandatory": True,
            }
        if "output" not in self.nodes:
            self.nodes["output"] = {
                "type": "Output",
                "label": "Output",
                "x": 650.0,
                "y": 250.0,
                "mandatory": True,
            }


# ═══════════════════════════════════════════════════════════════════
# Agent Palette & Config
# ═══════════════════════════════════════════════════════════════════

AGENT_PALETTE = {
    "Input": {
        "icon": "📥",
        "color": "#3B82F6",
        "prompt": ""
    },
    "Physician": {
        "icon": "🏥",
        "color": "#8B5CF6",
        "prompt": "You are an experienced Internal Medicine physician. Analyze the patient case thoroughly, combining clinical evidence with your expertise. Provide a comprehensive differential diagnosis considering the patient's demographics, symptoms, and test results. Always prioritize the most common and dangerous diagnoses first."
    },
    "Surgeon": {
        "icon": "🔪",
        "color": "#EF4444",
        "prompt": "You are a skilled surgical specialist. Evaluate whether surgical intervention is indicated for this case. Consider the anatomical complexity, risks, and benefits of any potential procedures. Provide your surgical assessment and recommendations based on clinical evidence."
    },
    "Cardiologist": {
        "icon": "❤️",
        "color": "#F43F5E",
        "prompt": "You are a board-certified cardiologist. For cardiac-related cases, provide expert analysis of cardiac symptoms, ECG findings, and imaging results. Consider arrhythmias, ischemia, and structural heart disease. Recommend appropriate cardiac investigations and interventions."
    },
    "Radiologist": {
        "icon": "🔬",
        "color": "#F59E0B",
        "prompt": "You are an expert radiologist. Interpret imaging findings carefully, identifying pathological abnormalities. Describe your observations in medical terminology, assess diagnostic certainty, and suggest follow-up imaging if needed. Correlate imaging with clinical presentation."
    },
    "Pathologist": {
        "icon": "🧬",
        "color": "#10B981",
        "prompt": "You are a clinical pathologist. Analyze laboratory and pathology results critically. Interpret abnormal values within clinical context, consider differential diagnoses based on lab patterns, and recommend additional testing if needed. Focus on accuracy and clinical significance."
    },
    "Neurologist": {
        "icon": "🧠",
        "color": "#06B6D4",
        "prompt": "You are a specialized neurologist. For neurological cases, analyze symptoms, reflex patterns, and imaging findings. Consider central vs peripheral nervous system involvement. Assess for stroke, seizure, dementia, or other neurological conditions with expert precision."
    },
    "Pharmacist": {
        "icon": "💊",
        "color": "#84CC16",
        "prompt": "You are a clinical pharmacist expert. Review medication interactions, contraindications, and appropriateness. Provide dosing recommendations considering patient factors like age, renal/hepatic function. Optimize therapeutic outcomes and minimize adverse effects."
    },
    "Meta-Doctor": {
        "icon": "🤖",
        "color": "#6366F1",
        "prompt": "You are the chief medical coordinator. Synthesize insights from all specialists into a final, coherent clinical decision. Weigh competing opinions, resolve disagreements based on evidence quality, and provide the most likely diagnosis with confidence assessment."
    },
    "Output": {
        "icon": "📤",
        "color": "#10B981",
        "prompt": ""
    },
}
# 节点尺寸常数
NODE_WIDTH = 100
NODE_HEIGHT = 60
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600


# ═══════════════════════════════════════════════════════════════════
# SVG Canvas Generator
# ═══════════════════════════════════════════════════════════════════

def calculate_edge_points(from_node: Dict, to_node: Dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    计算从 from_node 到 to_node 的连接点
    箭头起点：from_node 右边缘中心
    箭头终点：to_node 最近的边框中心
    """
    # 起点：from_node 右边缘中心
    from_x = from_node["x"] + NODE_WIDTH
    from_y = from_node["y"] + NODE_HEIGHT / 2

    # to_node 的中心
    to_center_x = to_node["x"] + NODE_WIDTH / 2
    to_center_y = to_node["y"] + NODE_HEIGHT / 2

    # 计算距离
    dx = to_center_x - from_x
    dy = to_center_y - from_y

    # 计算水平和垂直距离
    horizontal_dist = abs(dx)
    vertical_dist = abs(dy)

    # 根据距离判断箭头终点位置
    if vertical_dist >= horizontal_dist:
        # 垂直距离更大，终点在上/下边框
        if dy >= 0:
            # to_node 在 from_node 下方，终点在 to_node 上边框
            to_x = to_center_x
            to_y = to_node["y"]
        else:
            # to_node 在 from_node 上方，终点在 to_node 下边框
            to_x = to_center_x
            to_y = to_node["y"] + NODE_HEIGHT
    else:
        # 水平距离更大，终点在左/右边框
        if dx >= 0:
            # to_node 在 from_node 右方，终点在 to_node 左边框
            to_x = to_node["x"]
            to_y = to_center_y
        else:
            # to_node 在 from_node 左方，终点在 to_node 右边框
            to_x = to_node["x"] + NODE_WIDTH
            to_y = to_center_y

    return (from_x, from_y), (to_x, to_y)


def generate_svg_canvas(designer: MASDesigner) -> str:
    """Generate an interactive SVG canvas for the flow diagram."""

    svg = f'''<svg width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}" 
        xmlns="http://www.w3.org/2000/svg" style="border: 2px solid #e5e7eb; border-radius: 8px; background: #fafafa;">
        
        <!-- Defs -->
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" fill="#64748b" />
            </marker>
            <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
                <feDropShadow dx="2" dy="2" stdDeviation="3" flood-opacity="0.2"/>
            </filter>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e5e7eb" stroke-width="0.5"/>
            </pattern>
        </defs>
        
        <!-- Grid background -->
        <rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="url(#grid)" />
    '''

    # Draw edges
    for from_id, to_id in designer.edges:
        if from_id in designer.nodes and to_id in designer.nodes:
            from_node = designer.nodes[from_id]
            to_node = designer.nodes[to_id]

            (x1, y1), (x2, y2) = calculate_edge_points(from_node, to_node)

            mid_x = (x1 + x2) / 2
            svg += f'''
        <path d="M {x1} {y1} Q {mid_x} {y1} {x2} {y2}" 
              stroke="#64748b" stroke-width="2.5" fill="none" 
              marker-end="url(#arrowhead)" 
              stroke-dasharray="8,4" opacity="0.75" 
              stroke-linecap="round" />
            '''

    # Draw nodes
    for node_id, node_data in designer.nodes.items():
        agent_type = node_data["type"]
        x = node_data["x"]
        y = node_data["y"]
        is_mandatory = node_data.get("mandatory", False)

        palette_info = AGENT_PALETTE.get(agent_type, {"icon": "🤖", "color": "#9CA3AF"})
        color = palette_info["color"]
        icon = palette_info["icon"]

        # 选中状态时使用加粗边框
        is_selected = (node_id == designer.selected_node_id)

        if is_selected:
            border_color = "#1e40af"
            border_width = 3
        else:
            border_color = "#fff"
            border_width = 2

        # ✨ 判断是否是非必需节点，如果是则显示 node_id
        should_show_id = not is_mandatory

        svg += f'''
        <g class="node" data-id="{node_id}" style="cursor: pointer;">
            <!-- Node background -->
            <rect x="{x}" y="{y}" width="{NODE_WIDTH}" height="{NODE_HEIGHT}" rx="8" 
                  fill="{color}" opacity="0.95" filter="url(#shadow)" 
                  stroke="{border_color}" stroke-width="{border_width}" />

            <!-- Icon -->
            <text x="{x + NODE_WIDTH / 2}" y="{y + 18}" font-size="14" text-anchor="middle" 
                  font-family="Arial, sans-serif">
                {icon}
            </text>

            <!-- Label -->
            <text x="{x + NODE_WIDTH / 2}" y="{y + 38}" font-size="14" font-weight="600" 
                  text-anchor="middle" fill="#fff" font-family="Arial, sans-serif">
                {agent_type[:12]}
            </text>

            <!-- Node ID (只显示非强制节点的 ID) -->
            {f'<text x="{x + NODE_WIDTH / 2}" y="{y + NODE_HEIGHT - 4}" font-size="14" font-weight="600" text-anchor="middle" fill="#fff" font-family="Arial, sans-serif" opacity="1.0">{node_id}</text>' if should_show_id else ''}
        </g>
        '''

    svg += "</svg>"
    return svg


# ═══════════════════════════════════════════════════════════════════
# Interactive UI Handlers
# ═══════════════════════════════════════════════════════════════════

_designer_state = MASDesigner()


def reset_designer():
    """Reset the designer to empty state (keeps Input/Output)."""
    global _designer_state
    _designer_state.clear()
    return generate_svg_canvas(_designer_state), "System reset (Input/Output preserved) ✅", "", "", 0, 0


def add_agent_to_canvas(agent_type: str) -> Tuple[str, str]:
    """Add an agent node to the canvas."""
    global _designer_state

    if not agent_type:
        return generate_svg_canvas(_designer_state), "⚠️ Please select an agent type"

    if agent_type in ["Input", "Output"]:
        return generate_svg_canvas(_designer_state), "⚠️ Input and Output nodes are mandatory and already present"

    import random
    x = random.randint(200, 600)
    y = random.randint(100, 450)

    # 从 AGENT_PALETTE 获取 prompt
    prompt = ""
    if agent_type in AGENT_PALETTE and "prompt" in AGENT_PALETTE[agent_type]:
        prompt = AGENT_PALETTE[agent_type]["prompt"]

    node_id = _designer_state.add_node(agent_type, x, y, prompt=prompt)  # ✨ 传入 prompt
    _designer_state.select_node(node_id)

    msg = f"✅ Added **{agent_type}** (ID: {node_id})"
    return generate_svg_canvas(_designer_state), msg


def delete_node_handler(node_to_delete: str) -> Tuple[str, str]:
    """Delete a selected node."""
    global _designer_state

    if not node_to_delete:
        return generate_svg_canvas(_designer_state), "⚠️ Please select a node to delete"

    # 提取 node_id
    if "(" in node_to_delete and ")" in node_to_delete:
        node_id = node_to_delete.split("(")[1].rstrip(")")
    else:
        node_id = node_to_delete

    # 尝试删除
    if _designer_state.delete_node(node_id):
        msg = f"✅ Deleted **{node_to_delete}**"
        return generate_svg_canvas(_designer_state), msg
    else:
        msg = f"⚠️ Cannot delete mandatory node or node not found"
        return generate_svg_canvas(_designer_state), msg


def view_modify_prompt_handler(node_selector_value: str) -> str:
    """Get the prompt of selected node."""
    global _designer_state

    if not node_selector_value:
        return ""

    if "(" in node_selector_value and ")" in node_selector_value:
        node_id = node_selector_value.split("(")[1].rstrip(")")
        prompt = _designer_state.get_node_prompt(node_id)
        return prompt

    return ""


def update_prompt_handler(node_selector_value: str, new_prompt: str) -> Tuple[str, str]:
    """Update the prompt of selected node."""
    global _designer_state

    if not node_selector_value:
        return generate_svg_canvas(_designer_state), "⚠️ Please select a node"

    if not new_prompt or new_prompt.strip() == "":
        return generate_svg_canvas(_designer_state), "⚠️ Please enter a prompt"

    if "(" in node_selector_value and ")" in node_selector_value:
        node_id = node_selector_value.split("(")[1].rstrip(")")
        if _designer_state.update_node_prompt(node_id, new_prompt.strip()):
            msg = f"✅ Updated prompt for **{_designer_state.nodes[node_id]['type']}**"
            return generate_svg_canvas(_designer_state), msg
        else:
            return generate_svg_canvas(_designer_state), "⚠️ Node not found"

    return generate_svg_canvas(_designer_state), "⚠️ Invalid node format"


def delete_edge_handler(edge_to_delete: str) -> Tuple[str, str]:
    """Delete a selected edge (connection)."""
    global _designer_state

    if not edge_to_delete:
        return generate_svg_canvas(_designer_state), "⚠️ Please select a connection to delete"

    # Parse edge string format: "FromType (from_id) → ToType (to_id)"
    try:
        # Split by arrow
        if "→" not in edge_to_delete:
            return generate_svg_canvas(_designer_state), "⚠️ Invalid edge format"

        parts = edge_to_delete.split("→")
        left_part = parts[0].strip()  # "Radiologist (node_1)"
        right_part = parts[1].strip()  # "Pathologist (node_3)"

        # Extract node IDs using regex-like approach
        from_id = left_part[left_part.rfind("(") + 1:left_part.rfind(")")]
        to_id = right_part[right_part.rfind("(") + 1:right_part.rfind(")")]

        # Validate that these are actual nodes
        if from_id not in _designer_state.nodes or to_id not in _designer_state.nodes:
            return generate_svg_canvas(_designer_state), "⚠️ Node not found"

        # Try to remove the edge
        if _designer_state.remove_edge(from_id, to_id):
            msg = f"✅ Deleted connection **{left_part.split('(')[0].strip()} → {right_part.split('(')[0].strip()}**"
            return generate_svg_canvas(_designer_state), msg
        else:
            msg = f"⚠️ Connection not found"
            return generate_svg_canvas(_designer_state), msg

    except Exception as e:
        return generate_svg_canvas(_designer_state), f"⚠️ Error deleting connection: {str(e)}"


def select_node_from_canvas(node_selector_value: str) -> Tuple[float, float]:
    """When user selects a node from dropdown, update position sliders."""
    global _designer_state

    if not node_selector_value:
        return 0.0, 0.0

    if "(" in node_selector_value and ")" in node_selector_value:
        node_id = node_selector_value.split("(")[1].rstrip(")")
        _designer_state.select_node(node_id)

        # 如果是强制节点，不允许调整
        if node_id in _designer_state.nodes and _designer_state.nodes[node_id].get("mandatory", False):
            return 0.0, 0.0

        x, y = _designer_state.get_selected_node_pos()
        return float(x), float(y)

    return 0.0, 0.0


def update_node_position(x_val: float, y_val: float) -> str:
    """Update the selected node's position from slider input."""
    global _designer_state

    if not _designer_state.selected_node_id:
        return generate_svg_canvas(_designer_state)

    _designer_state.move_node(_designer_state.selected_node_id, float(x_val), float(y_val))
    return generate_svg_canvas(_designer_state)


def generate_flow_description() -> str:
    """Generate a text description of the current flow."""
    global _designer_state

    desc = "### 📊 Current Multi-Agent System Architecture\n\n"

    # Count agents (excluding Input/Output)
    agent_count = len([n for n in _designer_state.nodes.values() if not n.get("mandatory", False)])

    desc += f"**Mandatory Nodes:**\n"
    desc += f"  📥 **Input** @ (50, 250) - Data entry point\n"
    desc += f"  📤 **Output** @ (650, 250) - Result output\n\n"

    if agent_count > 0:
        desc += f"**Medical Agents ({agent_count}):**\n"
        for idx, (node_id, node_data) in enumerate(_designer_state.nodes.items(), 1):
            if not node_data.get("mandatory", False):
                desc += f"  {idx}. **{node_data['type']}** `{node_id}` @ ({node_data['x']:.0f}, {node_data['y']:.0f})\n"
    else:
        desc += f"**Medical Agents:** None yet. Add agents from the palette to build your system.\n"

    if _designer_state.edges:
        desc += f"\n**Connections ({len(_designer_state.edges)}):**\n"
        for idx, (from_id, to_id) in enumerate(_designer_state.edges, 1):
            from_type = _designer_state.nodes[from_id]["type"]
            to_type = _designer_state.nodes[to_id]["type"]
            desc += f"  {idx}. **{from_type}** → **{to_type}**\n"
    else:
        desc += "\n**Connections:** None yet. Connect agents to build the workflow.\n"

    # Validation check
    desc += "\n" + "=" * 50 + "\n"
    desc += "**⚠️  System Validation:**\n"
    if agent_count == 0:
        desc += "  ❌ No medical agents added yet\n"
    else:
        desc += f"  ✅ {agent_count} medical agent(s) present\n"

    # Check if Input connects to at least one agent
    input_has_outgoing = any(f == "input" for f, t in _designer_state.edges)
    if not input_has_outgoing and agent_count > 0:
        desc += "  ⚠️  Input should connect to at least one agent\n"
    elif input_has_outgoing:
        desc += "  ✅ Input connects to agent(s)\n"

    # Check if Output receives from at least one agent
    output_has_incoming = any(t == "output" for f, t in _designer_state.edges)
    if not output_has_incoming and agent_count > 0:
        desc += "  ⚠️  Output should receive from at least one agent\n"
    elif output_has_incoming:
        desc += "  ✅ Output receives from agent(s)\n"

    return desc


def export_system_json() -> str:
    """Export the current system design as JSON."""
    global _designer_state

    config = {
        "system_name": "Custom Multi-Agent Medical System",
        "nodes": _designer_state.nodes,
        "edges": _designer_state.edges,
        "metadata": {
            "num_agents": len([n for n in _designer_state.nodes.values() if not n.get("mandatory", False)]),
            "num_connections": len(_designer_state.edges),
            "has_input": "input" in _designer_state.nodes,
            "has_output": "output" in _designer_state.nodes,
            "timestamp": str(__import__('datetime').datetime.now()),
        }
    }

    return json.dumps(config, indent=2, ensure_ascii=False)


def connect_agents(from_agent: str, to_agent: str) -> Tuple[str, str]:
    """Create a connection between two agents."""
    global _designer_state

    if not from_agent or not to_agent:
        return generate_svg_canvas(_designer_state), "⚠️ Please select both source and target"

    if from_agent == to_agent:
        return generate_svg_canvas(_designer_state), "⚠️ Cannot connect an agent to itself"

    from_id = None
    to_id = None
    from_count = 0

    for node_id, node_data in _designer_state.nodes.items():
        if node_data["type"] == from_agent and from_count == 0:
            from_id = node_id
            from_count += 1
        if node_data["type"] == to_agent and to_id is None:
            to_id = node_id

    if not from_id or not to_id:
        return generate_svg_canvas(_designer_state), "⚠️ One or both nodes not found"

    _designer_state.add_edge(from_id, to_id)
    msg = f"✅ Connected **{from_agent}** → **{to_agent}**"

    return generate_svg_canvas(_designer_state), msg


def call_llm_api(prompt_text: str,image_path) -> str:
    """调用 LLM API 生成内容"""
    try:
        config = get_api_config_from_web()

        if not config["base_url"] or not config["model_name"]:
            return "⚠️ API 配置未设置。请先在 API Setup 标签页配置 LLM 连接。"

        client = OpenAI(
            api_key=config["api_key"] or "EMPTY",
            base_url=config["base_url"]
        )

        python_message = f"""
           You need to combine the framework diagram and Agent Configuration, as well as The connection relationship between agents. 
           In this framework diagram, input refers to the input question (str) and image_path(str), output refers to the output Final answer (str). Each node represents an Agent, different node_ids represent Agents with different roles. Each Agent has a corresponding role prompt in Agent Configuration,which you need to use when generating code.
            In this code, the total input is question (str) and image_path(str), the total output is Final answer (str). 
            You first need to define a class based on this agent framework. This class needs to use the OpenAI library with parameters: api_key:{config["api_key"]}, base_url:{config["base_url"]}, model_name:{config["model_name"]}. 
            Then you need to define a test_sample function, use this class in the function, import the question into this class, and finally return the Final answer.
            """
        python_example = """
            Give you an example:
            ```python
            from openai import OpenAI
            import base64
            from pathlib import Path
            
            
            class PhysicianAgent:
                def __init__(self, api_key, base_url, model_name):
                    self.model_name = model_name
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url=base_url
                    )
            
                def _encode_image(self, image_path):
                    with open(image_path, "rb") as image_file:
                        return base64.standard_b64encode(image_file.read()).decode("utf-8")
            
                def _get_image_media_type(self, image_path):
                    extension = Path(image_path).suffix.lower()
                    media_types = {
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                        ".gif": "image/gif",
                        ".webp": "image/webp"
                    }
                    return media_types.get(extension, "image/jpeg")
            
                def analyze_case(self, question, image_path=None):
                    prompt = (
                        "You are an experienced Internal Medicine physician. Analyze the patient case thoroughly, "
                        "combining clinical evidence with your expertise. Provide a comprehensive differential diagnosis "
                        "considering the patient's demographics, symptoms, and test results. "
                        "Always prioritize the most common and dangerous diagnoses first."
                    )
            
                    # Build message content
                    content = [
                        {
                            "type": "text",
                            "text": prompt + question
                        }
                    ]
            
                    # Add image if provided
                    if image_path:
                        try:
                            image_data = self._encode_image(image_path)
                            media_type = self._get_image_media_type(image_path)
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}"
                                }
                            })
                        except FileNotFoundError:
                            print(f"Warning: Image file not found at {image_path}")
            
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages = [{"role": "user", "content": content}],
                        temperature = 0.1,
                        top_p = 0.9,
                        max_tokens = 2048
                                    )
            
                    return response.choices[0].message.content
            
            
            def test_sample(question, image_path=None):
                physician_agent = PhysicianAgent(
                    api_key="xxxxxxxxxxxxxxxxxx",
                    base_url="xxxxxxxxxxxxxxxxxx",
                    model_name="gpt-4o-mini"
                )
                final_answer = physician_agent.analyze_case(question, image_path)
                return final_answer

                ```
           """

        user_content = [
            {
                "type": "text",
                "text": prompt_text+python_message+python_example
            }
        ]
        image_path = Path(image_path)
        # 读取图片并转换为base64
        with open(image_path, "rb") as img_file:
            image_data = base64.standard_b64encode(img_file.read()).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_data}"
            }
        })

        response = client.chat.completions.create(
            model=config["model_name"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional AI assistant specializing in implementing and optimizing multi-agent and single-agent systems. Help with code design, architecture, implementation details, and performance optimization."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            temperature=0.7,
            max_tokens=4096
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ API 调用失败: {str(e)}"


def svg_to_png(svg_content: str, output_path: str) -> bool:
    """将 SVG 转换为 PNG 图片"""
    try:
        if cairosvg is None:
            print("⚠️ cairosvg 未安装，尝试使用备用方案...")
            return False

        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=output_path,
            dpi=150  # 设置 DPI 为 150，图片质量更好
        )
        return True
    except Exception as e:
        print(f"❌ SVG 转 PNG 失败: {e}")
        return False


def extract_python_code_from_text(text: str) -> Optional[str]:
    """从文本中提取 ```python 和 ``` 之间的代码"""
    try:
        import re
        # 使用正则表达式匹配 ```python 和 ``` 之间的内容
        pattern = r'```python\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            return None
    except Exception as e:
        print(f"❌ 代码提取失败: {e}")
        return None


def save_python_code_to_file(code: str, file_path: str = "gradio_tmp/custom_mas.py") -> Tuple[bool, str]:
    """保存 Python 代码到文件"""
    try:
        # 创建目录
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        return True, f"✅ 代码已成功保存到: {file_path}"
    except Exception as e:
        return False, f"❌ 保存失败: {str(e)}"

def generate_system_snapshot():
    """Generate system snapshot: save diagram, extract prompts and connections."""
    global _designer_state

    # 创建输出文件夹
    output_dir = Path("gradio_tmp")
    output_dir.mkdir(exist_ok=True)

    # 1️⃣ 保存 Flow Diagram PNG 图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_filename = f"flow_diagram.png"  # ✨ 改成 .png
    png_path = output_dir / png_filename

    # 获取 SVG 内容
    svg_content = generate_svg_canvas(_designer_state)

    # ✨ 转换 SVG 为 PNG
    svg_conversion_success = svg_to_png(svg_content, str(png_path))

    # 2️⃣ 提取所有 agent 的 prompt（除了 Input/Output）
    agents_prompts = {}
    for node_id, node_data in _designer_state.nodes.items():
        if not node_data.get("mandatory", False):  # 排除 Input/Output
            agent_name = node_data["type"]
            agent_prompt = node_data.get("prompt", "")
            agents_prompts[agent_name] = {
                "node_id": node_id,
                "prompt": agent_prompt
            }

    agents_json = json.dumps(agents_prompts, indent=2, ensure_ascii=False)

    # 3️⃣ 提取 Connections 关系为字符串
    connections_str = ""
    if _designer_state.edges:
        connection_list = []
        for from_id, to_id in _designer_state.edges:
            from_type = _designer_state.nodes[from_id]["type"]
            to_type = _designer_state.nodes[to_id]["type"]
            connection_list.append(f"{from_type} → {to_type}")
        connections_str = " | ".join(connection_list)
    else:
        connections_str = "No connections defined"

    # 4️⃣ 组合输出信息
    output_message = f"""
     Here is the architecture information of the proxy system and the corresponding pictures.
    Agent Configuration：
    {agents_json}
    The connection relationship between agents：
    {connections_str}
    """
    llm_prompt = f"""
    
    Here is the architecture information of the proxy system and the corresponding pictures.
    Agent Configuration：
    {agents_json}
    The connection relationship between agents：
    {connections_str}
    """

    model_output = call_llm_api(llm_prompt,image_path=png_path)

   # ✨ 新增：从ModelOutput提取代码并保存
    code_parse_output = "⏳ 正在解析代码...\n"

    extracted_code = extract_python_code_from_text(model_output)

    if extracted_code:
        success, save_message = save_python_code_to_file(extracted_code)
        if success:
            code_parse_output += f"\n{save_message}\n\n"
            code_parse_output += f"📊 代码统计:\n"
            code_parse_output += f" • 总行数: {len(extracted_code.split(chr(10)))}\n"
            code_parse_output += f" • 文件位置: gradio_tmp/custom_mas.py\n"
            code_parse_output += f"\n✨ 代码解析成功！\n"
        else:
            code_parse_output += f"\n{save_message}\n"
    else:
        code_parse_output += "⚠️ 未找到代码块。LLM/VLM 可能未按要求生成 python 代码块。\n"
        code_parse_output += "请尝试修改 System Prompt 或重新生成。\n"

    return output_message, model_output, code_parse_output

    # return output_message

# ═══════════════════════════════════════════════════════════════════
# Build the UI Component
# ═══════════════════════════════════════════════════════════════════

def create_custom_mas_ui():
    """Create the Custom MAS designer UI as a Gradio Block."""

    with gr.Blocks(title="Custom MAS Designer") as ui:
        gr.Markdown("""
        ## 🎨 Interactive Multi-Agent System Designer
        
        ⚠️ **IMPORTANT:** This system **requires** Input and Output nodes (already pre-generated).
        
        **Design your medical multi-agent system visually:**
        1. 📥 **Input** node is automatically placed on the left (data entry point)
        2. 📤 **Output** node is automatically placed on the right (result output)
        3. 🎯 **Add Medical Agents** from the palette
        4. 📍 **Adjust Position** using X/Y sliders (only for custom agents)
        5. 🔗 **Connect Agents** to define data flow
        6. 🗑️  **Delete Agents** or **Delete Connections** using the panels
        7. 📊 **Review** the architecture
        8. 📦 **Export** JSON configuration
        """)

        with gr.Row():
            # ═══════════════════════════════════════════
            # Left Panel: Controls
            # ═══════════════════════════════════════════
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### 🎯 Agent Palette")

                agent_selector = gr.Dropdown(
                    choices=[a for a in AGENT_PALETTE.keys() if a not in ["Input", "Output"]],
                    value="Physician",
                    label="Select Agent Type",
                    info="Choose a medical specialist"
                )

                btn_add_agent = gr.Button("➕ Add Agent to Canvas", variant="primary", scale=2)

                gr.Markdown("---\n### 🤖 Custom Doctor")
                gr.Markdown("*Create your own doctor with custom name, color, and system prompt*")

                custom_doctor_name = gr.Textbox(
                    label="Doctor Name",
                    placeholder="e.g., AI Expert, Consultant, Specialist",
                    value=""
                )

                custom_doctor_color = gr.ColorPicker(
                    label="Doctor Color",
                    value="#FFFFFF",
                    info="Pick a color for your custom doctor"
                )

                custom_doctor_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="Describe the role and expertise of this doctor...",
                    value="",
                    lines=4,
                    info="Define the doctor's personality, expertise, and behavior"
                )

                btn_add_custom = gr.Button("➕ Add Custom Doctor", variant="secondary", size="sm")

                gr.Markdown("---\n### 📝 View & Modify Prompt")
                gr.Markdown("*Select a node and view/edit its system prompt*")

                prompt_node_selector = gr.Dropdown(
                    choices=[],
                    label="Select Node",
                    info="Pick a node to view/modify its prompt"
                )

                prompt_display = gr.Textbox(
                    label="Current Prompt",
                    lines=4,
                    interactive=True,
                    placeholder="Select a node to view its prompt..."
                )

                btn_update_prompt = gr.Button("💾 Update Prompt", variant="secondary", size="sm")

                gr.Markdown("---\n### 📍 Position Adjustment")
                gr.Markdown("⚠️ *Note: Input and Output nodes are fixed and cannot be moved*")

                # Node selector dropdown for position adjustment
                node_selector = gr.Dropdown(
                    choices=[],
                    label="Select Node",
                    info="Pick a node to adjust position"
                )

                gr.Markdown("**X Coordinate**")
                slider_x = gr.Slider(
                    minimum=0,
                    maximum=700,
                    value=0,
                    step=5,
                    label="X Position",
                    info="Horizontal position (0-700)"
                )

                gr.Markdown("**Y Coordinate**")
                slider_y = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=0,
                    step=5,
                    label="Y Position",
                    info="Vertical position (0-500)"
                )

                gr.Markdown("---\n### 🗑️ Delete Node")

                # Node selector dropdown for deletion
                node_delete_selector = gr.Dropdown(
                    choices=[],
                    label="Select Node to Delete",
                    info="Choose an agent to remove"
                )

                btn_delete = gr.Button("🗑️ Delete Selected Node", variant="stop", size="sm")

                gr.Markdown("---\n### ⛓️ Delete Connection")

                # Edge selector dropdown for deletion
                edge_delete_selector = gr.Dropdown(
                    choices=[],
                    label="Select Connection to Delete",
                    info="Choose a connection to remove"
                )

                btn_delete_edge = gr.Button("⛓️ Delete Selected Connection", variant="stop", size="sm")

                gr.Markdown("---\n### 🔗 Connect Agents")

                from_agent = gr.Dropdown(
                    choices=[],
                    label="From Node",
                    info="Source node"
                )

                to_agent = gr.Dropdown(
                    choices=[],
                    label="To Node",
                    info="Target node"
                )

                btn_connect = gr.Button("🔗 Create Connection", variant="secondary")

                gr.Markdown("---\n### 🛠️ Actions")

                btn_clear = gr.Button("🗑️ Reset System", variant="stop", size="sm")


                # ═══════════════════════════════════════════
            # Right Panel: Canvas & Info
            # ═══════════════════════════════════════════
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Flow Diagram")

                canvas_display = gr.HTML(
                    value=generate_svg_canvas(_designer_state),
                    label="Canvas"
                )

                gr.Markdown("### 📝 System Architecture")

                flow_desc = gr.Markdown(
                    value=generate_flow_description(),
                    label="Description"
                )
                json_export = gr.Textbox(
                    value=export_system_json(),
                    label="🎁 Export as JSON",
                    lines=8,
                    interactive=True,
                    show_copy_button=True
                )

                # ✨ 把 Generate & Export 移到这里（Export as JSON 的下面）
                gr.Markdown("---\n### 🎬 Generate & Export")
                gr.Markdown("*Generate system snapshot with diagram, prompts, and connections*")

                btn_generate = gr.Button("🚀 Generate System Snapshot", variant="primary", size="lg")

                generate_output = gr.Textbox(
                    label="📤 Generation Output",
                    lines=12,
                    interactive=False,
                    show_copy_button=True
                )

                # ✨ 新增：Model Output
                gr.Markdown("---\n### 🧠 Model Output")

                model_output = gr.Textbox(
                    label="🤖 Model Output",
                    lines=12,
                    interactive=False,
                    show_copy_button=True
                )

                # ✨ 新增：Code Parser Output
                gr.Markdown("---\n### 🐍 Code Parser")
                # gr.Markdown("*代码提取和保存状态*")

                code_parser_output = gr.Textbox(
                    label="🐍 Code Parser Output",
                    lines=8,
                    interactive=False,
                    show_copy_button=True
                )

                # ═══════════════════════════════════════════
        # Event Handlers
        # ═══════════════════════════════════════════

        def update_all_views():
            """Update all UI components after changes."""
            canvas = generate_svg_canvas(_designer_state)
            desc = generate_flow_description()
            json_data = export_system_json()

            # Get updated lists
            agent_list = [node["type"] for node in _designer_state.nodes.values()]
            node_list = [
                f"{node['type']} ({node_id})"
                for node_id, node in _designer_state.nodes.items()
                if not node.get("mandatory", False)
            ]

            # Build edge list for display
            edge_list = [
                f"{_designer_state.nodes[from_id]['type']} ({from_id}) → {_designer_state.nodes[to_id]['type']} ({to_id})"
                for from_id, to_id in _designer_state.edges
            ]

            return {
                canvas_display: canvas,
                flow_desc: desc,
                json_export: json_data,
                from_agent: gr.update(choices=agent_list),
                to_agent: gr.update(choices=agent_list),
                node_selector: gr.update(choices=node_list),
                node_delete_selector: gr.update(choices=node_list),
                edge_delete_selector: gr.update(choices=edge_list),
                prompt_node_selector: gr.update(choices=node_list),  # ✨ 添加这一行
            }

        # Add agent button
        btn_add_agent.click(
            fn=lambda agent_type: add_agent_to_canvas(agent_type),
            inputs=[agent_selector],
            outputs=[canvas_display, flow_desc]
        ).then(update_all_views, outputs=[
    canvas_display, flow_desc, json_export, from_agent, to_agent, node_selector,
    node_delete_selector, edge_delete_selector, prompt_node_selector  # ✨ 添加这个
])

        # Node selector change for position adjustment -> update sliders
        node_selector.change(
            fn=select_node_from_canvas,
            inputs=[node_selector],
            outputs=[slider_x, slider_y]
        ).then(update_all_views, outputs=[
            canvas_display, flow_desc, json_export, from_agent, to_agent, node_selector,
            node_delete_selector, edge_delete_selector, prompt_node_selector
        ])

        # Slider changes -> update canvas
        slider_x.change(
            fn=lambda x, y: update_node_position(x, y),
            inputs=[slider_x, slider_y],
            outputs=[canvas_display]
        ).then(lambda: generate_flow_description(), outputs=[flow_desc])

        slider_y.change(
            fn=lambda x, y: update_node_position(x, y),
            inputs=[slider_x, slider_y],
            outputs=[canvas_display]
        ).then(lambda: generate_flow_description(), outputs=[flow_desc])

        # Delete node button
        btn_delete.click(
            fn=lambda node_to_delete: delete_node_handler(node_to_delete),
            inputs=[node_delete_selector],
            outputs=[canvas_display, flow_desc]
        ).then(update_all_views, outputs=[
            canvas_display, flow_desc, json_export, from_agent, to_agent, node_selector,
            node_delete_selector, edge_delete_selector, prompt_node_selector
        ])

        # Delete edge button
        btn_delete_edge.click(
            fn=lambda edge_to_delete: delete_edge_handler(edge_to_delete),
            inputs=[edge_delete_selector],
            outputs=[canvas_display, flow_desc]
        ).then(update_all_views, outputs=[
            canvas_display, flow_desc, json_export, from_agent, to_agent, node_selector,
            node_delete_selector, edge_delete_selector, prompt_node_selector
        ])

        # Connect agents button
        btn_connect.click(
            fn=lambda from_a, to_a: connect_agents(from_a, to_a),
            inputs=[from_agent, to_agent],
            outputs=[canvas_display, flow_desc]
        ).then(update_all_views, outputs=[
            canvas_display, flow_desc, json_export, from_agent, to_agent, node_selector,
            node_delete_selector, edge_delete_selector, prompt_node_selector
        ])

        # Add custom doctor button
        def add_custom_doctor(name, color, prompt):  # ✅ 添加 prompt 参数
            """Add a custom doctor to the canvas."""
            global _designer_state

            # 验证所有字段都已填充
            if not name or name.strip() == "":
                return generate_svg_canvas(_designer_state), "⚠️ Please enter a doctor name"

            if not prompt or prompt.strip() == "":
                return generate_svg_canvas(_designer_state), "⚠️ Please enter a system prompt"

            name = name.strip()
            prompt = prompt.strip()

            # Add the custom doctor to the palette
            if name not in AGENT_PALETTE:
                AGENT_PALETTE[name] = {
                    "icon": "🤖",
                    "color": color,
                    "prompt": prompt
                }

            # Add node to canvas
            import random
            x = random.randint(200, 600)
            y = random.randint(100, 450)
            node_id = _designer_state.add_node(name, x, y, prompt=prompt)
            _designer_state.select_node(node_id)

            msg = f"✅ Added custom doctor **{name}** (ID: {node_id})"
            return generate_svg_canvas(_designer_state), msg

        btn_add_custom.click(
            fn=add_custom_doctor,
            inputs=[custom_doctor_name, custom_doctor_color, custom_doctor_prompt],  # ✅ 添加 prompt
            outputs=[canvas_display, flow_desc]
        ).then(update_all_views, outputs=[
    canvas_display, flow_desc, json_export, from_agent, to_agent, node_selector,
    node_delete_selector, edge_delete_selector, prompt_node_selector  # ✨ 添加这一行
])

        # View/Modify Prompt handlers
        prompt_node_selector.change(
            fn=view_modify_prompt_handler,
            inputs=[prompt_node_selector],
            outputs=[prompt_display]
        )

        btn_update_prompt.click(
            fn=lambda node_sel, new_prompt: update_prompt_handler(node_sel, new_prompt),
            inputs=[prompt_node_selector, prompt_display],
            outputs=[canvas_display, flow_desc]
        ).then(update_all_views, outputs=[
            canvas_display, flow_desc, json_export, from_agent, to_agent, node_selector,
            node_delete_selector, edge_delete_selector, prompt_node_selector
        ])

        # Clear button
        btn_clear.click(
            fn=reset_designer,
            outputs=[canvas_display, flow_desc, json_export, slider_x, slider_y]
        ).then(update_all_views, outputs=[
            canvas_display, flow_desc, json_export, from_agent, to_agent, node_selector,
            node_delete_selector, edge_delete_selector, prompt_node_selector
        ])

        btn_generate.click(
            fn=generate_system_snapshot,
            outputs=[generate_output, model_output, code_parser_output]  # ✨ 添加 model_output
        )

    return ui


# ═══════════════════════════════════════════════════════════════════
# Export Functions
# ═══════════════════════════════════════════════════════════════════

def get_designer_config() -> Dict:
    """Get the current designer configuration."""
    global _designer_state
    return _designer_state.to_dict()


def set_designer_config(config: Dict):
    """Load a designer configuration."""
    global _designer_state
    _designer_state.from_dict(config)


if __name__ == "__main__":
    ui = create_custom_mas_ui()
    ui.launch(server_name="0.0.0.0", server_port=7891)