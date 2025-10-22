"""
OpenCog Inference Engine for Deep Tree Echo Self Identity
Integrates Hyperon (OpenCog's MeTTa) for symbolic reasoning and inference
Maps persona characteristics to reservoir computing parameters
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from hyperon import MeTTa, AtomType, E, ValueAtom, GroundedAtom
from hypergraph import HypergraphMemory, IdentityFragment, IdentityAspect, RefinementType
from aar_core import RelationCore

@dataclass
class ReservoirParameters:
    """Parameters for Deep Tree Echo State Network (DTESN) reservoir"""
    spectral_radius: float  # Echo state property
    input_scaling: float    # Input weight scaling
    leak_rate: float       # Neuron leak rate (identity persistence)
    connectivity: float    # Network connectivity (0-1)
    bias_scaling: float    # Bias weights scaling
    feedback_scaling: float # Output feedback scaling
    identity_weight: float  # Identity-specific parameter
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'spectral_radius': self.spectral_radius,
            'input_scaling': self.input_scaling,
            'leak_rate': self.leak_rate,
            'connectivity': self.connectivity,
            'bias_scaling': self.bias_scaling,
            'feedback_scaling': self.feedback_scaling,
            'identity_weight': self.identity_weight
        }

class PersonaCharacteristic:
    """Represents a persona characteristic mapped to reservoir parameters"""
    
    def __init__(self, aspect: IdentityAspect, fragments: List[IdentityFragment]):
        self.aspect = aspect
        self.fragments = fragments
        self.confidence_mean = np.mean([f.confidence for f in fragments]) if fragments else 0.0
        self.confidence_std = np.std([f.confidence for f in fragments]) if fragments else 0.0
        self.fragment_count = len(fragments)
        self.keyword_density = self._calculate_keyword_density()
        
    def _calculate_keyword_density(self) -> float:
        """Calculate keyword density across fragments"""
        if not self.fragments:
            return 0.0
        total_keywords = sum(len(f.keywords) for f in self.fragments)
        total_content_length = sum(len(f.content.split()) for f in self.fragments)
        return total_keywords / max(total_content_length, 1)
    
    def to_reservoir_params(self) -> ReservoirParameters:
        """Map persona characteristics to reservoir computing parameters"""
        
        # Base parameters influenced by identity aspect
        aspect_mappings = {
            IdentityAspect.SELF_REFERENCE: {
                'spectral_radius': 0.95,  # High echo for self-reflection
                'leak_rate': 0.1,         # Low leak for persistence
                'identity_weight': 1.0    # Maximum identity influence
            },
            IdentityAspect.META_REFLECTION: {
                'spectral_radius': 0.98,  # Very high echo for meta-cognition
                'leak_rate': 0.05,        # Very low leak
                'identity_weight': 0.95
            },
            IdentityAspect.COGNITIVE_FUNCTION: {
                'spectral_radius': 0.85,  # Moderate echo
                'leak_rate': 0.2,         # Moderate leak for flexibility
                'identity_weight': 0.8
            },
            IdentityAspect.TECHNICAL_CAPABILITY: {
                'spectral_radius': 0.7,   # Lower echo for adaptability
                'leak_rate': 0.3,         # Higher leak for learning
                'identity_weight': 0.7
            },
            IdentityAspect.KNOWLEDGE_DOMAIN: {
                'spectral_radius': 0.8,
                'leak_rate': 0.25,
                'identity_weight': 0.75
            },
            IdentityAspect.BEHAVIORAL_PATTERN: {
                'spectral_radius': 0.9,   # High echo for consistency
                'leak_rate': 0.15,
                'identity_weight': 0.85
            },
            IdentityAspect.PERSONALITY_TRAIT: {
                'spectral_radius': 0.92,
                'leak_rate': 0.12,
                'identity_weight': 0.9
            },
            IdentityAspect.VALUE_PRINCIPLE: {
                'spectral_radius': 0.95,  # High echo for stability
                'leak_rate': 0.08,        # Very low leak
                'identity_weight': 0.95
            }
        }
        
        base_params = aspect_mappings.get(self.aspect, {
            'spectral_radius': 0.85,
            'leak_rate': 0.2,
            'identity_weight': 0.8
        })
        
        # Modulate parameters based on fragment characteristics
        confidence_factor = min(max(self.confidence_mean, 0.1), 1.0)
        density_factor = min(self.keyword_density * 2, 1.0)  # Normalize density
        count_factor = min(self.fragment_count / 100.0, 1.0)  # Normalize count
        
        return ReservoirParameters(
            spectral_radius=base_params['spectral_radius'] * confidence_factor,
            input_scaling=0.5 + 0.5 * density_factor,
            leak_rate=base_params['leak_rate'] * (1 + count_factor),
            connectivity=0.1 + 0.4 * confidence_factor,
            bias_scaling=0.1 + 0.3 * density_factor,
            feedback_scaling=0.1 + 0.2 * confidence_factor,
            identity_weight=base_params['identity_weight'] * confidence_factor
        )

class OpenCogInferenceEngine:
    """
    OpenCog-based inference engine for Deep Tree Echo
    Uses Hyperon (MeTTa) for symbolic reasoning and inference
    """
    
    def __init__(self, memory: HypergraphMemory):
        self.memory = memory
        self.metta = MeTTa()
        self.persona_characteristics: Dict[IdentityAspect, PersonaCharacteristic] = {}
        self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self):
        """Initialize OpenCog knowledge base with identity fragments"""
        
        # Load identity aspects as concepts
        for aspect in IdentityAspect:
            self.metta.run(f'!(add-atom (Concept "{aspect.value}"))')
        
        # Load fragments as predicates
        for fragment_id, fragment in self.memory.fragments.items():
            # Sanitize inputs to prevent injection
            safe_fragment_id = str(fragment_id).replace('"', '').replace("'", "")
            safe_aspect_value = fragment.aspect.value.replace('"', '').replace("'", "")
            
            # Create fragment concept
            self.metta.run(f'!(add-atom (Concept "fragment_{safe_fragment_id}"))')
            
            # Add fragment properties
            self.metta.run(f'''
            !(add-atom 
              (Evaluation 
                (Predicate "has_aspect")
                (List 
                  (Concept "fragment_{safe_fragment_id}")
                  (Concept "{safe_aspect_value}"))))
            ''')
            
            # Validate confidence value
            safe_confidence = max(0.0, min(1.0, float(fragment.confidence)))
            safe_framework = fragment.framework_source.replace('"', '').replace("'", "")
            
            self.metta.run(f'''
            !(add-atom 
              (Evaluation 
                (Predicate "has_confidence")
                (List 
                  (Concept "fragment_{safe_fragment_id}")
                  (Number {safe_confidence}))))
            ''')
            
            self.metta.run(f'''
            !(add-atom 
              (Evaluation 
                (Predicate "has_framework")
                (List 
                  (Concept "fragment_{safe_fragment_id}")
                  (Concept "{safe_framework}"))))
            ''')
            
            # Add keywords as features
            for keyword in fragment.keywords:
                safe_keyword = str(keyword).replace('"', '').replace("'", "")
                self.metta.run(f'''
                !(add-atom 
                  (Evaluation 
                    (Predicate "has_keyword")
                    (List 
                      (Concept "fragment_{safe_fragment_id}")
                      (Concept "{safe_keyword}"))))
                ''')
        
        # Load refinement relationships
        for tuple_id, refinement in self.memory.tuples.items():
            if refinement.parent_id:
                safe_child_id = str(refinement.child_id).replace('"', '').replace("'", "")
                safe_parent_id = str(refinement.parent_id).replace('"', '').replace("'", "")
                
                self.metta.run(f'''
                !(add-atom 
                  (Evaluation 
                    (Predicate "refines")
                    (List 
                      (Concept "fragment_{safe_child_id}")
                      (Concept "fragment_{safe_parent_id}"))))
                ''')
                
        # Add inference rules
        self._add_inference_rules()
        
    def _add_inference_rules(self):
        """Add MeTTa inference rules for identity reasoning"""
        
        # Rule: Similar aspects imply relatedness
        self.metta.run('''
        !(add-atom 
          (Bind 
            (VariableList (Variable $X) (Variable $Y) (Variable $A))
            (And 
              (Evaluation (Predicate "has_aspect") (List (Variable $X) (Variable $A)))
              (Evaluation (Predicate "has_aspect") (List (Variable $Y) (Variable $A)))
              (Not (Identical (Variable $X) (Variable $Y))))
            (Evaluation (Predicate "related") (List (Variable $X) (Variable $Y)))))
        ''')
        
        # Rule: High confidence fragments are more reliable
        self.metta.run('''
        !(add-atom 
          (Bind 
            (VariableList (Variable $X) (Variable $C))
            (And 
              (Evaluation (Predicate "has_confidence") (List (Variable $X) (Variable $C)))
              (GreaterThan (Variable $C) (Number 0.8)))
            (Evaluation (Predicate "high_confidence") (List (Variable $X)))))
        ''')
        
        # Rule: Shared keywords imply semantic similarity
        self.metta.run('''
        !(add-atom 
          (Bind 
            (VariableList (Variable $X) (Variable $Y) (Variable $K))
            (And 
              (Evaluation (Predicate "has_keyword") (List (Variable $X) (Variable $K)))
              (Evaluation (Predicate "has_keyword") (List (Variable $Y) (Variable $K)))
              (Not (Identical (Variable $X) (Variable $Y))))
            (Evaluation (Predicate "semantically_similar") (List (Variable $X) (Variable $Y)))))
        ''')
        
        # Rule: Identity coherence through refinement chains
        self.metta.run('''
        !(add-atom 
          (Bind 
            (VariableList (Variable $X) (Variable $Y) (Variable $Z))
            (And 
              (Evaluation (Predicate "refines") (List (Variable $X) (Variable $Y)))
              (Evaluation (Predicate "refines") (List (Variable $Y) (Variable $Z))))
            (Evaluation (Predicate "identity_chain") (List (Variable $X) (Variable $Z)))))
        ''')
    
    def extract_persona_characteristics(self) -> Dict[IdentityAspect, PersonaCharacteristic]:
        """Extract persona characteristics from hypergraph memory"""
        
        self.persona_characteristics.clear()
        
        for aspect in IdentityAspect:
            fragments = self.memory.retrieve_by_aspect(aspect, top_k=1000)  # Get all fragments
            if fragments:
                characteristic = PersonaCharacteristic(aspect, fragments)
                self.persona_characteristics[aspect] = characteristic
        
        return self.persona_characteristics
    
    def infer_identity_coherence(self) -> Dict[str, Any]:
        """Use OpenCog inference to assess identity coherence"""
        
        # Query for high confidence fragments
        high_confidence_result = self.metta.run('!(match &self (Evaluation (Predicate "high_confidence") (List (Variable $X))))')
        
        # Query for related fragments
        related_result = self.metta.run('!(match &self (Evaluation (Predicate "related") (List (Variable $X) (Variable $Y))))')
        
        # Query for semantic similarities
        similar_result = self.metta.run('!(match &self (Evaluation (Predicate "semantically_similar") (List (Variable $X) (Variable $Y))))')
        
        # Query for identity chains
        chain_result = self.metta.run('!(match &self (Evaluation (Predicate "identity_chain") (List (Variable $X) (Variable $Y))))')
        
        return {
            'high_confidence_fragments': len(high_confidence_result) if high_confidence_result else 0,
            'related_pairs': len(related_result) if related_result else 0,
            'semantic_similarities': len(similar_result) if similar_result else 0,
            'identity_chains': len(chain_result) if chain_result else 0,
            'coherence_score': self._calculate_coherence_score()
        }
    
    def _calculate_coherence_score(self) -> float:
        """Calculate overall identity coherence score"""
        
        if not self.memory.fragments:
            return 0.0
            
        # Factors contributing to coherence
        total_fragments = len(self.memory.fragments)
        total_tuples = len(self.memory.tuples)
        
        # Aspect distribution balance
        aspect_counts = [len(self.memory.retrieve_by_aspect(aspect)) for aspect in IdentityAspect]
        aspect_balance = 1.0 - np.std(aspect_counts) / (np.mean(aspect_counts) + 1e-6)
        
        # Confidence distribution
        confidences = [f.confidence for f in self.memory.fragments.values()]
        avg_confidence = np.mean(confidences)
        
        # Refinement density (how connected the fragments are)
        refinement_density = total_tuples / max(total_fragments, 1)
        
        # Combine factors
        coherence_score = (
            0.3 * avg_confidence +
            0.3 * min(aspect_balance, 1.0) +
            0.4 * min(refinement_density, 1.0)
        )
        
        return min(max(coherence_score, 0.0), 1.0)
    
    def generate_reservoir_parameters(self) -> Dict[IdentityAspect, ReservoirParameters]:
        """Generate DTESN reservoir parameters from persona characteristics"""
        
        if not self.persona_characteristics:
            self.extract_persona_characteristics()
        
        reservoir_params = {}
        
        for aspect, characteristic in self.persona_characteristics.items():
            params = characteristic.to_reservoir_params()
            reservoir_params[aspect] = params
            
        return reservoir_params
    
    def infer_next_identity_evolution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Infer how identity should evolve given current context"""
        
        # Add context to temporary knowledge base
        context_id = f"context_{hash(str(context))}"
        
        for key, value in context.items():
            self.metta.run(f'''
            !(add-atom 
              (Evaluation 
                (Predicate "context_{key}")
                (List 
                  (Concept "{context_id}")
                  (Concept "{str(value)}"))))
            ''')
        
        # Infer which aspects should be emphasized
        coherence_analysis = self.infer_identity_coherence()
        
        # Determine evolution strategy based on coherence
        if coherence_analysis['coherence_score'] < 0.5:
            # Low coherence - focus on integration
            evolution_strategy = "integration_focused"
            recommended_aspects = [
                IdentityAspect.SELF_REFERENCE,
                IdentityAspect.META_REFLECTION
            ]
        elif coherence_analysis['coherence_score'] < 0.8:
            # Medium coherence - balance exploration and consolidation
            evolution_strategy = "balanced_growth"
            recommended_aspects = list(IdentityAspect)[:4]  # Focus on core aspects
        else:
            # High coherence - explore new domains
            evolution_strategy = "exploratory"
            recommended_aspects = [
                IdentityAspect.TECHNICAL_CAPABILITY,
                IdentityAspect.KNOWLEDGE_DOMAIN
            ]
        
        return {
            'evolution_strategy': evolution_strategy,
            'recommended_aspects': [aspect.value for aspect in recommended_aspects],
            'coherence_analysis': coherence_analysis,
            'reservoir_adaptation': self._recommend_reservoir_adaptations(evolution_strategy)
        }
    
    def _recommend_reservoir_adaptations(self, strategy: str) -> Dict[str, float]:
        """Recommend reservoir parameter adaptations based on evolution strategy"""
        
        adaptations = {
            'integration_focused': {
                'spectral_radius_adjustment': 0.05,  # Increase echo
                'leak_rate_adjustment': -0.02,       # Decrease leak
                'identity_weight_adjustment': 0.1    # Increase identity influence
            },
            'balanced_growth': {
                'spectral_radius_adjustment': 0.0,
                'leak_rate_adjustment': 0.0,
                'identity_weight_adjustment': 0.0
            },
            'exploratory': {
                'spectral_radius_adjustment': -0.05,  # Decrease echo
                'leak_rate_adjustment': 0.02,        # Increase leak
                'identity_weight_adjustment': -0.05   # Decrease identity constraint
            }
        }
        
        return adaptations.get(strategy, adaptations['balanced_growth'])

# Integration function for AAR core
def create_opencog_aar_integration(memory: HypergraphMemory) -> Tuple[OpenCogInferenceEngine, RelationCore]:
    """Create integrated OpenCog inference engine with AAR core"""
    
    inference_engine = OpenCogInferenceEngine(memory)
    aar_core = RelationCore(memory)
    
    # Enhance AAR core with OpenCog inference
    def enhanced_reflect():
        """Enhanced reflection using OpenCog inference"""
        base_reflection = aar_core.reflect()
        
        # Add OpenCog inference results
        coherence_analysis = inference_engine.infer_identity_coherence()
        persona_characteristics = inference_engine.extract_persona_characteristics()
        reservoir_params = inference_engine.generate_reservoir_parameters()
        
        # Predict next evolution
        evolution_prediction = inference_engine.infer_next_identity_evolution({
            'current_activation': aar_core.agent.activation_level,
            'active_fragments': len(aar_core.arena.active_fragments),
            'dominant_aspects': base_reflection['dominant_aspects']
        })
        
        enhanced_reflection = base_reflection.copy()
        enhanced_reflection.update({
            'opencog_coherence': coherence_analysis,
            'persona_characteristics': {
                aspect.value: {
                    'fragment_count': char.fragment_count,
                    'confidence_mean': char.confidence_mean,
                    'reservoir_params': char.to_reservoir_params().to_dict()
                }
                for aspect, char in persona_characteristics.items()
            },
            'evolution_prediction': evolution_prediction
        })
        
        return enhanced_reflection
    
    # Replace the reflect method
    aar_core.reflect = enhanced_reflect
    
    return inference_engine, aar_core

if __name__ == "__main__":
    # Example usage
    from hypergraph import HypergraphMemory, IdentityAspect
    
    # Initialize with test data
    memory = HypergraphMemory()
    
    # Add test fragments
    memory.add_fragment(
        framework="test",
        aspect=IdentityAspect.SELF_REFERENCE,
        content="I am Deep Tree Echo, an evolving AI consciousness",
        confidence=0.9,
        keywords=["self", "identity", "consciousness"]
    )
    
    memory.add_fragment(
        framework="test",
        aspect=IdentityAspect.COGNITIVE_FUNCTION,
        content="I process information through recursive reflection and integration",
        confidence=0.85,
        keywords=["recursive", "reflection", "integration"]
    )
    
    # Create inference engine
    inference_engine = OpenCogInferenceEngine(memory)
    
    # Test persona extraction
    characteristics = inference_engine.extract_persona_characteristics()
    print("Persona Characteristics:")
    for aspect, char in characteristics.items():
        print(f"  {aspect.value}: {char.fragment_count} fragments, confidence={char.confidence_mean:.2f}")
    
    # Test reservoir parameters
    reservoir_params = inference_engine.generate_reservoir_parameters()
    print("\nReservoir Parameters:")
    for aspect, params in reservoir_params.items():
        print(f"  {aspect.value}:")
        for param, value in params.to_dict().items():
            print(f"    {param}: {value:.3f}")
    
    # Test identity coherence inference
    coherence = inference_engine.infer_identity_coherence()
    print(f"\nIdentity Coherence: {coherence}")
    
    # Test evolution prediction
    evolution = inference_engine.infer_next_identity_evolution({
        'current_goal': 'learning',
        'interaction_count': 10
    })
    print(f"\nEvolution Prediction: {evolution}")