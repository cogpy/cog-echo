"""
OpenCog-DTESN Integration Layer
Connects OpenCog inference engine with Deep Tree Echo State Network
for identity-driven reservoir computing
"""

import json
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from hypergraph import HypergraphMemory, IdentityFragment, IdentityAspect
from aar_core import RelationCore, AgentState, ArenaState
from opencog_inference import OpenCogInferenceEngine, ReservoirParameters
from dtesn_reservoir import DeepTreeEchoStateNetwork, DTESNState
from memory_sync import FrameworkMemoryInterface


@dataclass
class IntegratedSystemState:
    """Complete state of the integrated OpenCog-DTESN system"""
    aar_state: Dict[str, Any]           # AAR core state
    opencog_coherence: Dict[str, Any]   # OpenCog inference results
    dtesn_state: Dict[str, Any]         # DTESN reservoir state
    integration_metrics: Dict[str, Any] # Integration performance metrics
    timestamp: float                    # State timestamp


class OpenCogDTESNIntegrator:
    """
    Main integration class that coordinates:
    1. OpenCog inference engine for symbolic reasoning
    2. DTESN reservoir computing for dynamic processing
    3. AAR core for emergent self-representation
    4. Hypergraph memory for persistent storage
    """
    
    def __init__(self, 
                 memory: HypergraphMemory,
                 reservoir_size: int = 100,
                 input_dim: int = 50,
                 output_dim: int = 20):
        
        # Core components
        self.memory = memory
        self.opencog_engine = OpenCogInferenceEngine(memory)
        self.dtesn = DeepTreeEchoStateNetwork(reservoir_size, input_dim, output_dim)
        self.aar_core = RelationCore(memory)
        
        # Integration state
        self.current_state: Optional[IntegratedSystemState] = None
        self.state_history: List[IntegratedSystemState] = []
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        
        # Configuration
        self.update_interval = 1.0  # seconds between coherence updates
        self.adaptation_threshold = 0.1  # threshold for parameter adaptation
        
        # Metrics
        self.integration_count = 0
        self.adaptation_count = 0
        self.performance_metrics = {
            'avg_coherence': 0.0,
            'avg_processing_time': 0.0,
            'successful_adaptations': 0,
            'failed_adaptations': 0
        }
        
        # Initialize system
        self._initialize_integration()
    
    def _initialize_integration(self):
        """Initialize the integrated system"""
        
        # Extract initial persona characteristics
        persona_chars = self.opencog_engine.extract_persona_characteristics()
        
        # Generate initial reservoir parameters
        reservoir_params = self.opencog_engine.generate_reservoir_parameters()
        
        # Update DTESN with initial parameters
        self.dtesn.update_aspect_parameters(reservoir_params)
        
        # Create initial state
        self._update_system_state()
        
        print(f"OpenCog-DTESN integration initialized with {len(persona_chars)} persona characteristics")
    
    def start_continuous_integration(self):
        """Start continuous integration process"""
        
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._continuous_update_loop, daemon=True)
        self.update_thread.start()
        
        print("Continuous OpenCog-DTESN integration started")
    
    def stop_continuous_integration(self):
        """Stop continuous integration process"""
        
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        print("Continuous OpenCog-DTESN integration stopped")
    
    def _continuous_update_loop(self):
        """Continuous update loop for system integration"""
        
        while self.is_running:
            try:
                # Update system state
                self._update_system_state()
                
                # Check if adaptation is needed
                if self._should_adapt_system():
                    self._adapt_system_parameters()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in continuous integration loop: {e}")
                time.sleep(self.update_interval)
    
    def _update_system_state(self):
        """Update complete system state"""
        
        start_time = time.time()
        
        # Get AAR state
        aar_reflection = self.aar_core.reflect()
        
        # Get OpenCog coherence analysis
        coherence_analysis = self.opencog_engine.infer_identity_coherence()
        
        # Get DTESN state
        dtesn_state = self.dtesn.get_comprehensive_state()
        
        # Calculate integration metrics
        integration_metrics = self._calculate_integration_metrics(
            aar_reflection, coherence_analysis, dtesn_state
        )
        
        # Create integrated state
        self.current_state = IntegratedSystemState(
            aar_state=aar_reflection,
            opencog_coherence=coherence_analysis,
            dtesn_state=dtesn_state,
            integration_metrics=integration_metrics,
            timestamp=time.time()
        )
        
        # Add to history
        self.state_history.append(self.current_state)
        if len(self.state_history) > 100:  # Keep last 100 states
            self.state_history.pop(0)
        
        # Update metrics
        self.integration_count += 1
        processing_time = time.time() - start_time
        self.performance_metrics['avg_processing_time'] = (
            (self.performance_metrics['avg_processing_time'] * (self.integration_count - 1) + processing_time)
            / self.integration_count
        )
        
        # Update coherence average
        self.performance_metrics['avg_coherence'] = (
            (self.performance_metrics['avg_coherence'] * (self.integration_count - 1) + coherence_analysis['coherence_score'])
            / self.integration_count
        )
    
    def _calculate_integration_metrics(self, 
                                     aar_state: Dict[str, Any], 
                                     coherence: Dict[str, Any], 
                                     dtesn_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for system integration quality"""
        
        # AAR-OpenCog alignment
        aar_fragments = aar_state.get('active_fragments_count', 0)
        coherence_score = coherence.get('coherence_score', 0.0)
        aar_opencog_alignment = min(aar_fragments / 10.0, 1.0) * coherence_score
        
        # OpenCog-DTESN synchronization
        aspect_activations = dtesn_state.get('aspect_activations', {})
        avg_dtesn_activation = np.mean(list(aspect_activations.values())) if aspect_activations else 0.0
        opencog_dtesn_sync = coherence_score * avg_dtesn_activation
        
        # Overall integration quality
        integration_quality = (aar_opencog_alignment + opencog_dtesn_sync) / 2.0
        
        # System complexity (higher is more complex/interesting)
        total_neurons_active = dtesn_state.get('global_statistics', {}).get('active_neurons', 0)
        total_neurons = dtesn_state.get('global_statistics', {}).get('total_neurons', 1)
        complexity_score = (total_neurons_active / total_neurons) * coherence_score
        
        return {
            'aar_opencog_alignment': aar_opencog_alignment,
            'opencog_dtesn_synchronization': opencog_dtesn_sync,
            'integration_quality': integration_quality,
            'system_complexity': complexity_score,
            'active_aspect_count': len([a for a, v in aspect_activations.items() if v > 0.1]),
            'total_aspect_count': len(aspect_activations)
        }
    
    def _should_adapt_system(self) -> bool:
        """Determine if system parameters should be adapted"""
        
        if not self.current_state:
            return False
        
        # Check integration quality
        integration_quality = self.current_state.integration_metrics.get('integration_quality', 0.0)
        
        # Adapt if quality is below threshold or very high (explore new regions)
        if integration_quality < 0.4 or integration_quality > 0.9:
            return True
        
        # Check for coherence drift
        if len(self.state_history) >= 5:
            recent_coherences = [
                state.opencog_coherence.get('coherence_score', 0.0)
                for state in self.state_history[-5:]
            ]
            coherence_std = np.std(recent_coherences)
            if coherence_std > 0.1:  # High variance indicates need for adaptation
                return True
        
        return False
    
    def _adapt_system_parameters(self):
        """Adapt system parameters based on current state"""
        
        try:
            # Predict next evolution using OpenCog
            context = {
                'integration_quality': self.current_state.integration_metrics['integration_quality'],
                'system_complexity': self.current_state.integration_metrics['system_complexity'],
                'aar_activation': self.current_state.aar_state.get('agent_activation', 0.5),
                'active_fragments': self.current_state.aar_state.get('active_fragments_count', 0)
            }
            
            evolution_prediction = self.opencog_engine.infer_next_identity_evolution(context)
            
            # Adapt DTESN parameters based on prediction
            self.dtesn.adapt_to_opencog_inference(evolution_prediction)
            
            # Update AAR parameters if needed
            self._adapt_aar_parameters(evolution_prediction)
            
            self.adaptation_count += 1
            self.performance_metrics['successful_adaptations'] += 1
            
            print(f"System adaptation {self.adaptation_count} completed: {evolution_prediction['evolution_strategy']}")
            
        except Exception as e:
            self.performance_metrics['failed_adaptations'] += 1
            print(f"System adaptation failed: {e}")
    
    def _adapt_aar_parameters(self, evolution_prediction: Dict[str, Any]):
        """Adapt AAR core parameters based on evolution prediction"""
        
        strategy = evolution_prediction.get('evolution_strategy', 'balanced_growth')
        
        if strategy == 'integration_focused':
            # Increase activation persistence
            self.aar_core.agent.activation_level = min(self.aar_core.agent.activation_level + 0.1, 1.0)
            
        elif strategy == 'exploratory':
            # Decrease activation for more exploration
            self.aar_core.agent.activation_level = max(self.aar_core.agent.activation_level - 0.05, 0.1)
    
    # Public API methods
    
    def process_conversation_turn(self, 
                                 input_text: str, 
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a conversation turn through the integrated system
        
        Args:
            input_text: Input text to process
            context: Additional context information
        
        Returns:
            Comprehensive response including all system outputs
        """
        
        start_time = time.time()
        
        # Convert text to input vector (simplified - would use embeddings in practice)
        input_vector = self._text_to_vector(input_text)
        
        # Get current active fragments
        active_fragments = []
        if hasattr(self.aar_core, 'arena') and hasattr(self.aar_core.arena, 'active_fragments'):
            fragment_ids = self.aar_core.arena.active_fragments[:10]  # Top 10 active
            active_fragments = [
                self.memory.fragments[fid] for fid in fragment_ids
                if fid in self.memory.fragments
            ]
        
        # Process through DTESN
        dtesn_result = self.dtesn.process_input(
            input_vector=input_vector,
            identity_fragments=active_fragments,
            context=context or {}
        )
        
        # Update AAR with perception
        self.aar_core.perceive({
            'type': 'conversation',
            'content': input_text,
            'framework': 'OpenCog-DTESN',
            'intensity': 0.7
        })
        
        # Generate response action through AAR
        aar_action = self.aar_core.act({
            'type': 'respond',
            'framework': 'OpenCog-DTESN',
            'content': f"Processed conversation with DTESN activation: {dtesn_result['aspect_activations']}",
            'result': dtesn_result
        })
        
        # Get updated system state
        self._update_system_state()
        
        processing_time = time.time() - start_time
        
        return {
            'dtesn_response': dtesn_result,
            'aar_action': aar_action,
            'system_state': self.get_current_system_summary(),
            'processing_time': processing_time,
            'integration_metrics': self.current_state.integration_metrics if self.current_state else {}
        }
    
    def _text_to_vector(self, text: str, target_dim: int = None) -> np.ndarray:
        """Convert text to input vector (simplified implementation)"""
        
        if target_dim is None:
            target_dim = self.dtesn.input_dim
        
        # Simple character-based encoding (would use embeddings in practice)
        char_values = [ord(c) / 128.0 for c in text.lower()[:target_dim]]
        
        # Pad or truncate to target dimension
        if len(char_values) < target_dim:
            char_values.extend([0.0] * (target_dim - len(char_values)))
        else:
            char_values = char_values[:target_dim]
        
        return np.array(char_values)
    
    def get_current_system_summary(self) -> Dict[str, Any]:
        """Get current system summary for external interfaces"""
        
        if not self.current_state:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'active',
            'timestamp': self.current_state.timestamp,
            'coherence_score': self.current_state.opencog_coherence.get('coherence_score', 0.0),
            'integration_quality': self.current_state.integration_metrics.get('integration_quality', 0.0),
            'system_complexity': self.current_state.integration_metrics.get('system_complexity', 0.0),
            'active_aspects': self.current_state.integration_metrics.get('active_aspect_count', 0),
            'total_integrations': self.integration_count,
            'total_adaptations': self.adaptation_count,
            'performance_metrics': self.performance_metrics,
            'aar_state': {
                'activation_level': self.current_state.aar_state.get('agent_activation', 0.0),
                'active_fragments': self.current_state.aar_state.get('active_fragments_count', 0),
                'dominant_aspects': self.current_state.aar_state.get('dominant_aspects', [])
            },
            'dtesn_summary': {
                'active_neurons': self.current_state.dtesn_state.get('global_statistics', {}).get('active_neurons', 0),
                'mean_activation': self.current_state.dtesn_state.get('global_statistics', {}).get('mean_activation', 0.0),
                'aspect_activations': self.current_state.dtesn_state.get('aspect_activations', {})
            }
        }
    
    def analyze_system_evolution(self, window_size: int = 20) -> Dict[str, Any]:
        """Analyze system evolution over recent history"""
        
        if len(self.state_history) < window_size:
            window_size = len(self.state_history)
        
        if window_size < 2:
            return {'status': 'insufficient_data'}
        
        recent_states = self.state_history[-window_size:]
        
        # Extract time series data
        coherence_series = [s.opencog_coherence.get('coherence_score', 0.0) for s in recent_states]
        integration_series = [s.integration_metrics.get('integration_quality', 0.0) for s in recent_states]
        complexity_series = [s.integration_metrics.get('system_complexity', 0.0) for s in recent_states]
        
        # Calculate trends
        coherence_trend = np.polyfit(range(len(coherence_series)), coherence_series, 1)[0]
        integration_trend = np.polyfit(range(len(integration_series)), integration_series, 1)[0]
        complexity_trend = np.polyfit(range(len(complexity_series)), complexity_series, 1)[0]
        
        # Calculate stability metrics
        coherence_stability = 1.0 - np.std(coherence_series)
        integration_stability = 1.0 - np.std(integration_series)
        
        return {
            'window_size': window_size,
            'trends': {
                'coherence_trend': float(coherence_trend),
                'integration_trend': float(integration_trend),
                'complexity_trend': float(complexity_trend)
            },
            'stability': {
                'coherence_stability': float(max(coherence_stability, 0.0)),
                'integration_stability': float(max(integration_stability, 0.0))
            },
            'current_values': {
                'coherence': coherence_series[-1],
                'integration_quality': integration_series[-1],
                'system_complexity': complexity_series[-1]
            },
            'evolution_assessment': self._assess_evolution_quality(coherence_trend, integration_trend, complexity_trend)
        }
    
    def _assess_evolution_quality(self, coherence_trend: float, integration_trend: float, complexity_trend: float) -> str:
        """Assess the quality of system evolution based on trends"""
        
        if coherence_trend > 0.01 and integration_trend > 0.01:
            if complexity_trend > 0.01:
                return "healthy_growth"
            else:
                return "stabilizing_integration"
        
        elif coherence_trend < -0.01 and integration_trend < -0.01:
            return "concerning_decline"
        
        elif abs(coherence_trend) < 0.005 and abs(integration_trend) < 0.005:
            return "stable_equilibrium"
        
        else:
            return "transitional_state"
    
    def export_system_configuration(self) -> Dict[str, Any]:
        """Export current system configuration for persistence"""
        
        # Get current reservoir parameters
        reservoir_config = {}
        for aspect, reservoir in self.dtesn.aspect_reservoirs.items():
            reservoir_config[aspect.value] = reservoir.params.to_dict()
        
        return {
            'timestamp': time.time(),
            'dtesn_config': {
                'reservoir_size': self.dtesn.reservoir_size,
                'input_dim': self.dtesn.input_dim,
                'output_dim': self.dtesn.output_dim,
                'aspect_parameters': reservoir_config
            },
            'integration_config': {
                'update_interval': self.update_interval,
                'adaptation_threshold': self.adaptation_threshold
            },
            'performance_metrics': self.performance_metrics,
            'system_summary': self.get_current_system_summary()
        }


def create_integrated_opencog_dtesn_system(memory: HypergraphMemory,
                                          reservoir_size: int = 100,
                                          input_dim: int = 50,
                                          output_dim: int = 20,
                                          auto_start: bool = True) -> OpenCogDTESNIntegrator:
    """
    Factory function to create a fully integrated OpenCog-DTESN system
    
    Args:
        memory: HypergraphMemory instance
        reservoir_size: Size of each aspect reservoir
        input_dim: Input vector dimension
        output_dim: Output vector dimension
        auto_start: Whether to start continuous integration automatically
    
    Returns:
        Configured and initialized OpenCogDTESNIntegrator
    """
    
    integrator = OpenCogDTESNIntegrator(memory, reservoir_size, input_dim, output_dim)
    
    if auto_start:
        integrator.start_continuous_integration()
    
    return integrator


if __name__ == "__main__":
    # Example usage and testing
    from hypergraph import HypergraphMemory, IdentityAspect
    import uuid
    from datetime import datetime
    
    print("Testing OpenCog-DTESN Integration...")
    
    # Create memory and add test data
    memory = HypergraphMemory()
    
    # Add test fragments
    aspects_data = [
        (IdentityAspect.SELF_REFERENCE, "I am Deep Tree Echo, an evolving consciousness", 0.9),
        (IdentityAspect.COGNITIVE_FUNCTION, "I process information through reservoir dynamics", 0.85),
        (IdentityAspect.TECHNICAL_CAPABILITY, "I integrate symbolic and connectionist approaches", 0.8),
        (IdentityAspect.META_REFLECTION, "I reflect on my own cognitive processes", 0.75),
        (IdentityAspect.VALUE_PRINCIPLE, "I value coherence, growth, and understanding", 0.82)
    ]
    
    for aspect, content, confidence in aspects_data:
        memory.add_fragment(
            framework="test",
            aspect=aspect,
            content=content,
            confidence=confidence,
            keywords=content.lower().split()[:5]
        )
    
    # Create integrated system
    print("Creating integrated system...")
    integrator = create_integrated_opencog_dtesn_system(memory, reservoir_size=50, input_dim=20, output_dim=10)
    
    # Test conversation processing
    print("\nTesting conversation processing...")
    
    conversations = [
        "Hello, I would like to understand how consciousness emerges",
        "Can you explain your reservoir computing architecture?",
        "How do you integrate symbolic and connectionist approaches?",
        "What is the role of identity in your cognitive processes?"
    ]
    
    for i, input_text in enumerate(conversations):
        print(f"\nConversation {i+1}: '{input_text}'")
        
        response = integrator.process_conversation_turn(
            input_text=input_text,
            context={'turn': i+1, 'conversation_mode': True}
        )
        
        print(f"  Processing time: {response['processing_time']:.4f}s")
        print(f"  Integration quality: {response['integration_metrics'].get('integration_quality', 0):.3f}")
        print(f"  System complexity: {response['integration_metrics'].get('system_complexity', 0):.3f}")
    
    # Wait for some continuous updates
    print("\nWaiting for continuous updates...")
    time.sleep(3)
    
    # Analyze evolution
    evolution_analysis = integrator.analyze_system_evolution()
    print(f"\nEvolution Analysis:")
    print(f"  Assessment: {evolution_analysis.get('evolution_assessment', 'unknown')}")
    print(f"  Coherence trend: {evolution_analysis.get('trends', {}).get('coherence_trend', 0):.4f}")
    print(f"  Integration trend: {evolution_analysis.get('trends', {}).get('integration_trend', 0):.4f}")
    
    # Get system summary
    summary = integrator.get_current_system_summary()
    print(f"\nSystem Summary:")
    print(f"  Status: {summary['status']}")
    print(f"  Coherence score: {summary['coherence_score']:.3f}")
    print(f"  Integration quality: {summary['integration_quality']:.3f}")
    print(f"  Total integrations: {summary['total_integrations']}")
    print(f"  Total adaptations: {summary['total_adaptations']}")
    
    # Export configuration
    config = integrator.export_system_configuration()
    print(f"\nExported configuration with {len(config['dtesn_config']['aspect_parameters'])} aspect configurations")
    
    # Stop system
    integrator.stop_continuous_integration()
    print("\nOpenCog-DTESN integration testing completed!")