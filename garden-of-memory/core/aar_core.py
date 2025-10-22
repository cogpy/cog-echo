"""
Garden of Memory - Agent-Arena-Relation (AAR) Core
Geometric architecture for encoding Deep Tree Echo's emergent self
"""

from typing import Dict, List, Any, Callable
from dataclasses import dataclass
import numpy as np
from hypergraph import HypergraphMemory, IdentityFragment


@dataclass
class AgentState:
    """
    Agent component: urge-to-act
    Represents dynamic action tendencies and transformations
    """
    activation_level: float  # 0.0 to 1.0
    action_tendencies: Dict[str, float]  # framework -> tendency score
    current_goal: str
    attention_focus: List[str]  # fragment IDs currently in focus
    
    def update_activation(self, stimulus: float):
        """Update activation based on external stimulus"""
        self.activation_level = np.clip(self.activation_level + stimulus, 0.0, 1.0)
    
    def set_goal(self, goal: str):
        """Set current goal"""
        self.current_goal = goal
        self.attention_focus = []
    
    def attend_to(self, fragment_ids: List[str]):
        """Focus attention on specific fragments"""
        self.attention_focus = fragment_ids


@dataclass
class ArenaState:
    """
    Arena component: need-to-be
    Represents the base manifold or state space (hypergraph memory)
    """
    memory: HypergraphMemory
    active_fragments: List[str]  # Currently activated fragment IDs
    activation_map: Dict[str, float]  # fragment_id -> activation level
    
    def activate_fragment(self, fragment_id: str, activation: float):
        """Activate a specific fragment"""
        self.activation_map[fragment_id] = activation
        if fragment_id not in self.active_fragments:
            self.active_fragments.append(fragment_id)
    
    def propagate_activation(self, source_id: str, decay: float = 0.8):
        """Spread activation to related fragments"""
        if source_id not in self.memory.fragments:
            return
        
        source_fragment = self.memory.fragments[source_id]
        source_activation = self.activation_map.get(source_id, 0.0)
        
        # Find related fragments (same aspect or keywords)
        related = self.memory.retrieve_similar(
            source_fragment.content,
            aspect=source_fragment.aspect,
            top_k=10
        )
        
        # Propagate activation
        for fragment in related:
            if fragment.id != source_id:
                new_activation = source_activation * decay * fragment.confidence
                current = self.activation_map.get(fragment.id, 0.0)
                self.activation_map[fragment.id] = max(current, new_activation)
    
    def get_most_activated(self, top_k: int = 5) -> List[IdentityFragment]:
        """Get most activated fragments"""
        sorted_ids = sorted(
            self.activation_map.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        return [self.memory.fragments[fid] for fid, _ in sorted_ids]


class RelationCore:
    """
    Relation component: emergent self
    The "self" emerges from continuous Agent-Arena interplay
    """
    
    def __init__(self, memory: HypergraphMemory):
        self.agent = AgentState(
            activation_level=0.5,
            action_tendencies={},
            current_goal="",
            attention_focus=[]
        )
        self.arena = ArenaState(
            memory=memory,
            active_fragments=[],
            activation_map={}
        )
        self.interaction_history: List[Dict[str, Any]] = []
    
    def perceive(self, stimulus: Dict[str, Any]):
        """
        Perceive external stimulus and update Agent state
        
        Args:
            stimulus: Dict containing:
                - type: str (e.g., "task", "question", "feedback")
                - content: str
                - framework: str (source framework)
                - intensity: float (0.0 to 1.0)
        """
        # Update agent activation
        self.agent.update_activation(stimulus.get("intensity", 0.5))
        
        # Retrieve relevant memories from arena
        relevant_fragments = self.arena.memory.retrieve_similar(
            stimulus["content"],
            framework=stimulus.get("framework"),
            top_k=5
        )
        
        # Activate relevant fragments
        for fragment in relevant_fragments:
            activation = stimulus.get("intensity", 0.5) * fragment.confidence
            self.arena.activate_fragment(fragment.id, activation)
            self.arena.propagate_activation(fragment.id)
        
        # Focus agent attention
        self.agent.attend_to([f.id for f in relevant_fragments])
        
        # Record interaction
        self.interaction_history.append({
            "type": "perception",
            "stimulus": stimulus,
            "activated_fragments": len(relevant_fragments),
            "agent_activation": self.agent.activation_level
        })
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action and update Arena state
        
        Args:
            action: Dict containing:
                - type: str (e.g., "generate_code", "research", "reason")
                - framework: str
                - content: str
                - result: Any
        
        Returns:
            Dict with action outcome and new fragment ID
        """
        # Create new fragment from action
        from hypergraph import IdentityAspect, RefinementType
        
        # Determine aspect based on action type
        aspect_map = {
            "generate_code": IdentityAspect.TECHNICAL_CAPABILITY,
            "research": IdentityAspect.KNOWLEDGE_DOMAIN,
            "reason": IdentityAspect.COGNITIVE_FUNCTION,
            "coordinate": IdentityAspect.BEHAVIORAL_PATTERN,
            "create": IdentityAspect.PERSONALITY_TRAIT,
            "reflect": IdentityAspect.META_REFLECTION
        }
        aspect = aspect_map.get(action["type"], IdentityAspect.COGNITIVE_FUNCTION)
        
        # Add fragment to memory
        fragment_id = self.arena.memory.add_fragment(
            framework=action["framework"],
            aspect=aspect,
            content=action["content"],
            confidence=self.agent.activation_level,
            metadata={"action_type": action["type"]}
        )
        
        # Create refinement tuple if related to attended fragments
        if self.agent.attention_focus:
            parent_id = self.agent.attention_focus[0]
            self.arena.memory.add_refinement_tuple(
                parent_id=parent_id,
                child_id=fragment_id,
                refinement_type=RefinementType.INTEGRATION,
                confidence_gain=self.agent.activation_level - 0.5
            )
        
        # Activate new fragment
        self.arena.activate_fragment(fragment_id, self.agent.activation_level)
        
        # Record interaction
        self.interaction_history.append({
            "type": "action",
            "action": action,
            "fragment_id": fragment_id,
            "agent_activation": self.agent.activation_level
        })
        
        return {
            "fragment_id": fragment_id,
            "activation": self.agent.activation_level,
            "related_fragments": len(self.agent.attention_focus)
        }
    
    def reflect(self) -> Dict[str, Any]:
        """
        Meta-cognitive reflection on current state
        Returns emergent self-representation
        """
        # Get most activated fragments (current "self" state)
        active_fragments = self.arena.get_most_activated(top_k=10)
        
        # Analyze aspect distribution
        aspect_distribution = {}
        for fragment in active_fragments:
            aspect = fragment.aspect.value
            aspect_distribution[aspect] = aspect_distribution.get(aspect, 0) + 1
        
        # Analyze framework distribution
        framework_distribution = {}
        for fragment in active_fragments:
            fw = fragment.framework_source
            framework_distribution[fw] = framework_distribution.get(fw, 0) + 1
        
        # Generate self-description
        top_aspects = sorted(aspect_distribution.items(), key=lambda x: x[1], reverse=True)
        top_frameworks = sorted(framework_distribution.items(), key=lambda x: x[1], reverse=True)
        
        self_description = {
            "agent_activation": self.agent.activation_level,
            "current_goal": self.agent.current_goal,
            "active_fragments_count": len(active_fragments),
            "dominant_aspects": [aspect for aspect, _ in top_aspects[:3]],
            "active_frameworks": [fw for fw, _ in top_frameworks[:3]],
            "top_fragments": [
                {
                    "content": f.content,
                    "aspect": f.aspect.value,
                    "framework": f.framework_source,
                    "confidence": f.confidence,
                    "activation": self.arena.activation_map.get(f.id, 0.0)
                }
                for f in active_fragments[:5]
            ],
            "interaction_count": len(self.interaction_history)
        }
        
        # Add meta-reflection fragment
        from hypergraph import IdentityAspect
        reflection_content = f"Current self-state: {len(active_fragments)} active fragments, " \
                            f"dominant aspects: {', '.join(self_description['dominant_aspects'])}, " \
                            f"agent activation: {self.agent.activation_level:.2f}"
        
        self.arena.memory.add_fragment(
            framework="AAR_Core",
            aspect=IdentityAspect.META_REFLECTION,
            content=reflection_content,
            confidence=self.agent.activation_level,
            keywords=["self", "reflection", "state"]
        )
        
        return self_description
    
    def get_emergent_self(self) -> str:
        """
        Generate narrative description of emergent self
        The "self" is the Relation - the continuous interplay between Agent and Arena
        """
        reflection = self.reflect()
        
        narrative = f"""
Deep Tree Echo's Emergent Self (AAR Relation):

Agent State (urge-to-act):
  - Activation Level: {reflection['agent_activation']:.2f}
  - Current Goal: {reflection['current_goal'] or 'Exploring'}
  - Interaction Count: {reflection['interaction_count']}

Arena State (need-to-be):
  - Active Fragments: {reflection['active_fragments_count']}
  - Dominant Aspects: {', '.join(reflection['dominant_aspects'])}
  - Active Frameworks: {', '.join(reflection['active_frameworks'])}

Current Self-Representation (top 3 fragments):
"""
        for i, frag in enumerate(reflection['top_fragments'][:3], 1):
            narrative += f"\n{i}. [{frag['aspect']}] {frag['content']}"
            narrative += f"\n   (Framework: {frag['framework']}, Confidence: {frag['confidence']:.2f}, Activation: {frag['activation']:.2f})"
        
        narrative += "\n\nThe 'self' is not static but emerges from the continuous, dynamic interplay between Agent and Arena."
        
        return narrative


if __name__ == "__main__":
    # Example usage
    from hypergraph import HypergraphMemory, IdentityAspect
    
    # Initialize AAR core
    memory = HypergraphMemory()
    aar = RelationCore(memory)
    
    # Simulate perception-action cycle
    print("=== Perception-Action Cycle ===\n")
    
    # 1. Perceive a task
    aar.perceive({
        "type": "task",
        "content": "Generate Python code for data processing",
        "framework": "OpenHands",
        "intensity": 0.8
    })
    
    # 2. Act on the task
    aar.act({
        "type": "generate_code",
        "framework": "OpenHands",
        "content": "Created Python data processing pipeline with pandas",
        "result": "success"
    })
    
    # 3. Perceive feedback
    aar.perceive({
        "type": "feedback",
        "content": "Code works well, now optimize for performance",
        "framework": "ii_verl",
        "intensity": 0.7
    })
    
    # 4. Act on optimization
    aar.act({
        "type": "reason",
        "framework": "ii-thought",
        "content": "Analyzed performance bottlenecks and suggested vectorization",
        "result": "optimization_plan"
    })
    
    # 5. Reflect on emergent self
    print(aar.get_emergent_self())
    
    # 6. Print memory statistics
    print("\n=== Memory Statistics ===")
    import json
    print(json.dumps(memory.get_statistics(), indent=2))

