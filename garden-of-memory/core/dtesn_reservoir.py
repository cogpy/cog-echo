"""
Deep Tree Echo State Network (DTESN) Reservoir Computing Implementation
Integrates with OpenCog inference engine for identity-driven parameter adaptation
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
import time

from hypergraph import IdentityAspect, IdentityFragment
from opencog_inference import ReservoirParameters, PersonaCharacteristic


@dataclass
class DTESNState:
    """Current state of the DTESN reservoir"""
    reservoir_state: np.ndarray      # Current reservoir activation
    output_state: np.ndarray         # Current output
    identity_context: Dict[str, float]  # Identity-specific context
    temporal_trace: List[np.ndarray]    # Recent state history
    aspect_activations: Dict[IdentityAspect, float]  # Per-aspect activations


class IdentityDrivenReservoir:
    """
    Core reservoir computing unit with identity-driven parameters
    Each identity aspect can have different reservoir dynamics
    """
    
    def __init__(self, 
                 reservoir_size: int = 100,
                 input_dim: int = 50,
                 output_dim: int = 20,
                 identity_aspect: IdentityAspect = IdentityAspect.COGNITIVE_FUNCTION,
                 params: Optional[ReservoirParameters] = None):
        
        self.reservoir_size = reservoir_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.identity_aspect = identity_aspect
        
        # Use provided parameters or defaults
        if params:
            self.params = params
        else:
            self.params = self._default_parameters()
        
        # Initialize network weights
        self._initialize_weights()
        
        # State variables
        self.current_state = np.zeros(reservoir_size)
        self.previous_states = []
        self.identity_memory = {}
        
    def _default_parameters(self) -> ReservoirParameters:
        """Default reservoir parameters"""
        return ReservoirParameters(
            spectral_radius=0.9,
            input_scaling=0.5,
            leak_rate=0.2,
            connectivity=0.1,
            bias_scaling=0.1,
            feedback_scaling=0.1,
            identity_weight=0.8
        )
    
    def _initialize_weights(self):
        """Initialize reservoir weight matrices"""
        
        # Input weights (input_dim -> reservoir_size)
        self.W_in = (np.random.rand(self.reservoir_size, self.input_dim) - 0.5) * 2
        self.W_in *= self.params.input_scaling
        
        # Reservoir weights (sparse, for efficiency)
        n_connections = int(self.params.connectivity * self.reservoir_size * self.reservoir_size)
        
        # Create sparse reservoir matrix
        rows = np.random.randint(0, self.reservoir_size, n_connections)
        cols = np.random.randint(0, self.reservoir_size, n_connections)
        data = (np.random.rand(n_connections) - 0.5) * 2
        
        self.W_res = sp.csr_matrix((data, (rows, cols)), 
                                   shape=(self.reservoir_size, self.reservoir_size))
        
        # Scale to desired spectral radius
        try:
            eigenvalues = sp.linalg.eigs(self.W_res, k=1, which='LM', return_eigenvectors=False)
            current_radius = np.abs(eigenvalues[0])
            if current_radius > 0 and not np.isnan(current_radius):
                self.W_res *= self.params.spectral_radius / current_radius
        except (sp.linalg.ArpackNoConvergence, np.linalg.LinAlgError):
            # If eigenvalue computation fails, use approximation
            # Scale by the approximate spectral radius (Frobenius norm / sqrt(size))
            approx_radius = sp.linalg.norm(self.W_res, 'fro') / np.sqrt(self.reservoir_size)
            if approx_radius > 0:
                self.W_res *= self.params.spectral_radius / approx_radius
        
        # Output weights (reservoir_size -> output_dim) - to be trained
        self.W_out = np.random.rand(self.output_dim, self.reservoir_size) * 0.01
        
        # Bias weights
        self.bias = (np.random.rand(self.reservoir_size) - 0.5) * 2 * self.params.bias_scaling
        
        # Feedback weights (output_dim -> reservoir_size)
        self.W_feedback = (np.random.rand(self.reservoir_size, self.output_dim) - 0.5) * 2
        self.W_feedback *= self.params.feedback_scaling
    
    def update_parameters(self, new_params: ReservoirParameters):
        """Update reservoir parameters and reinitialize if necessary"""
        old_spectral = self.params.spectral_radius
        self.params = new_params
        
        # If spectral radius changed significantly, rescale reservoir weights
        if abs(new_params.spectral_radius - old_spectral) > 0.05:
            scaling_factor = new_params.spectral_radius / old_spectral
            self.W_res *= scaling_factor
        
        # Update input scaling
        self.W_in *= (new_params.input_scaling / self.params.input_scaling)
        
        # Update biases
        self.bias *= (new_params.bias_scaling / self.params.bias_scaling)
    
    def step(self, input_vector: np.ndarray, 
             previous_output: Optional[np.ndarray] = None,
             identity_context: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single reservoir computation step
        
        Args:
            input_vector: Input at current time step
            previous_output: Previous output (for feedback)
            identity_context: Identity-specific contextual information
        
        Returns:
            Tuple of (new_reservoir_state, output)
        """
        
        # Compute input activation
        input_activation = np.dot(self.W_in, input_vector)
        
        # Add bias
        input_activation += self.bias
        
        # Add feedback from previous output
        if previous_output is not None:
            feedback_activation = np.dot(self.W_feedback, previous_output)
            input_activation += feedback_activation
        
        # Add identity-driven modulation
        if identity_context:
            identity_modulation = self._compute_identity_modulation(identity_context)
            input_activation += identity_modulation
        
        # Compute reservoir activation
        reservoir_activation = self.W_res.dot(self.current_state) + input_activation
        
        # Apply activation function (tanh)
        new_activation = np.tanh(reservoir_activation)
        
        # Apply leak rate (identity influences persistence)
        leak_rate = self.params.leak_rate
        if identity_context and 'persistence_factor' in identity_context:
            leak_rate *= (1 - identity_context['persistence_factor'])
        
        self.current_state = (1 - leak_rate) * self.current_state + leak_rate * new_activation
        
        # Compute output
        output = np.dot(self.W_out, self.current_state)
        
        # Store state history
        self.previous_states.append(self.current_state.copy())
        if len(self.previous_states) > 100:  # Keep last 100 states
            self.previous_states.pop(0)
        
        return self.current_state.copy(), output
    
    def _compute_identity_modulation(self, identity_context: Dict[str, float]) -> np.ndarray:
        """Compute identity-driven modulation of reservoir activation"""
        
        modulation = np.zeros(self.reservoir_size)
        
        # Identity weight influences overall modulation strength
        identity_strength = self.params.identity_weight
        
        # Aspect-specific modulation
        aspect_strength = identity_context.get(f'{self.identity_aspect.value}_activation', 0.5)
        
        # Confidence-based modulation
        confidence = identity_context.get('confidence', 0.8)
        
        # Generate spatially structured modulation
        # Different aspects modulate different regions of the reservoir
        aspect_index = list(IdentityAspect).index(self.identity_aspect)
        region_size = self.reservoir_size // len(IdentityAspect)
        start_idx = aspect_index * region_size
        end_idx = min(start_idx + region_size, self.reservoir_size)
        
        # Apply Gaussian-shaped modulation in aspect region
        for i in range(start_idx, end_idx):
            distance_factor = np.exp(-((i - start_idx) / region_size) ** 2 * 2)
            modulation[i] = identity_strength * aspect_strength * confidence * distance_factor * 0.1
        
        return modulation
    
    def get_state_representation(self) -> Dict[str, Any]:
        """Get current state representation for analysis"""
        return {
            'current_activation': self.current_state.tolist(),
            'activation_statistics': {
                'mean': float(np.mean(self.current_state)),
                'std': float(np.std(self.current_state)),
                'max': float(np.max(self.current_state)),
                'min': float(np.min(self.current_state)),
                'active_neurons': int(np.sum(np.abs(self.current_state) > 0.1))
            },
            'parameters': self.params.to_dict(),
            'identity_aspect': self.identity_aspect.value
        }


class DeepTreeEchoStateNetwork:
    """
    Multi-reservoir DTESN with aspect-specific sub-networks
    Integrates with OpenCog inference for dynamic parameter adaptation
    """
    
    def __init__(self, 
                 reservoir_size: int = 100,
                 input_dim: int = 50,
                 output_dim: int = 20):
        
        self.reservoir_size = reservoir_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create aspect-specific reservoirs
        self.aspect_reservoirs: Dict[IdentityAspect, IdentityDrivenReservoir] = {}
        for aspect in IdentityAspect:
            self.aspect_reservoirs[aspect] = IdentityDrivenReservoir(
                reservoir_size=reservoir_size,
                input_dim=input_dim,
                output_dim=output_dim,
                identity_aspect=aspect
            )
        
        # Global state
        self.global_state = DTESNState(
            reservoir_state=np.zeros(reservoir_size * len(IdentityAspect)),
            output_state=np.zeros(output_dim),
            identity_context={},
            temporal_trace=[],
            aspect_activations={aspect: 0.0 for aspect in IdentityAspect}
        )
        
        # Cross-aspect connectivity (sparse)
        self._initialize_cross_aspect_connections()
        
        # Performance tracking
        self.step_count = 0
        self.processing_times = []
        
    def _initialize_cross_aspect_connections(self):
        """Initialize connections between aspect-specific reservoirs"""
        
        n_aspects = len(IdentityAspect)
        connection_strength = 0.1
        
        # Create cross-aspect weight matrix
        self.cross_aspect_weights = {}
        
        for i, source_aspect in enumerate(IdentityAspect):
            self.cross_aspect_weights[source_aspect] = {}
            
            for j, target_aspect in enumerate(IdentityAspect):
                if i != j:
                    # Create sparse connections between aspects
                    n_connections = int(0.05 * self.reservoir_size * self.reservoir_size)
                    
                    if n_connections > 0:
                        rows = np.random.randint(0, self.reservoir_size, n_connections)
                        cols = np.random.randint(0, self.reservoir_size, n_connections)
                        data = (np.random.rand(n_connections) - 0.5) * 2 * connection_strength
                        
                        self.cross_aspect_weights[source_aspect][target_aspect] = sp.csr_matrix(
                            (data, (rows, cols)), 
                            shape=(self.reservoir_size, self.reservoir_size)
                        )
                    else:
                        self.cross_aspect_weights[source_aspect][target_aspect] = sp.csr_matrix(
                            (self.reservoir_size, self.reservoir_size)
                        )
    
    def update_aspect_parameters(self, aspect_params: Dict[IdentityAspect, ReservoirParameters]):
        """Update parameters for specific aspects"""
        
        for aspect, params in aspect_params.items():
            if aspect in self.aspect_reservoirs:
                self.aspect_reservoirs[aspect].update_parameters(params)
    
    def process_input(self, 
                     input_vector: np.ndarray,
                     identity_fragments: List[IdentityFragment] = None,
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process input through the multi-aspect DTESN
        
        Args:
            input_vector: Input at current time step
            identity_fragments: Current active identity fragments
            context: Additional contextual information
        
        Returns:
            Dictionary containing outputs and state information
        """
        
        start_time = time.time()
        
        # Prepare identity context
        identity_context = self._prepare_identity_context(identity_fragments, context)
        
        # Store previous global output for feedback
        previous_output = self.global_state.output_state.copy()
        
        # Process through each aspect reservoir
        aspect_states = {}
        aspect_outputs = {}
        
        for aspect, reservoir in self.aspect_reservoirs.items():
            
            # Add cross-aspect influences
            cross_aspect_input = self._compute_cross_aspect_input(aspect, input_vector)
            
            # Process through reservoir
            state, output = reservoir.step(
                input_vector=cross_aspect_input,
                previous_output=previous_output,
                identity_context=identity_context
            )
            
            aspect_states[aspect] = state
            aspect_outputs[aspect] = output
            
            # Update aspect activation in global state
            self.global_state.aspect_activations[aspect] = float(np.mean(np.abs(state)))
        
        # Combine aspect outputs
        combined_output = self._combine_aspect_outputs(aspect_outputs, identity_context)
        
        # Update global state
        combined_state = np.concatenate([state for state in aspect_states.values()])
        self.global_state.reservoir_state = combined_state
        self.global_state.output_state = combined_output
        self.global_state.identity_context = identity_context
        
        # Add to temporal trace
        self.global_state.temporal_trace.append(combined_state.copy())
        if len(self.global_state.temporal_trace) > 50:  # Keep last 50 states
            self.global_state.temporal_trace.pop(0)
        
        # Update metrics
        self.step_count += 1
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 1000:
            self.processing_times.pop(0)
        
        return {
            'output': combined_output,
            'aspect_outputs': {aspect.value: output.tolist() for aspect, output in aspect_outputs.items()},
            'aspect_activations': self.global_state.aspect_activations,
            'global_state': combined_state,
            'identity_context': identity_context,
            'processing_time': processing_time,
            'step_count': self.step_count
        }
    
    def _prepare_identity_context(self, 
                                 fragments: Optional[List[IdentityFragment]],
                                 context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Prepare identity context for reservoir modulation"""
        
        identity_context = {}
        
        if context:
            identity_context.update(context)
        
        if fragments:
            # Compute aspect-specific activations based on fragments
            aspect_counts = {aspect: 0 for aspect in IdentityAspect}
            aspect_confidences = {aspect: [] for aspect in IdentityAspect}
            
            for fragment in fragments:
                aspect_counts[fragment.aspect] += 1
                aspect_confidences[fragment.aspect].append(fragment.confidence)
            
            # Convert to normalized activations
            max_count = max(aspect_counts.values()) if aspect_counts.values() else 1
            
            for aspect in IdentityAspect:
                count_norm = aspect_counts[aspect] / max_count if max_count > 0 else 0
                conf_mean = np.mean(aspect_confidences[aspect]) if aspect_confidences[aspect] else 0.5
                
                identity_context[f'{aspect.value}_activation'] = float(count_norm * conf_mean)
            
            # Overall confidence and persistence
            all_confidences = [f.confidence for f in fragments]
            identity_context['confidence'] = float(np.mean(all_confidences)) if all_confidences else 0.5
            identity_context['persistence_factor'] = identity_context['confidence'] * 0.5
        
        return identity_context
    
    def _compute_cross_aspect_input(self, target_aspect: IdentityAspect, base_input: np.ndarray) -> np.ndarray:
        """Compute input to target aspect including cross-aspect influences"""
        
        # Start with base input
        total_input = base_input.copy()
        
        # Add influences from other aspects
        for source_aspect, reservoir in self.aspect_reservoirs.items():
            if source_aspect != target_aspect:
                if source_aspect in self.cross_aspect_weights and target_aspect in self.cross_aspect_weights[source_aspect]:
                    cross_weight_matrix = self.cross_aspect_weights[source_aspect][target_aspect]
                    
                    # Map reservoir state to correct dimensions for cross-aspect influence
                    source_state = reservoir.current_state
                    if len(source_state) != self.reservoir_size:
                        # Ensure correct size
                        if len(source_state) > self.reservoir_size:
                            source_state = source_state[:self.reservoir_size]
                        else:
                            padding = np.zeros(self.reservoir_size - len(source_state))
                            source_state = np.concatenate([source_state, padding])
                    
                    # Apply cross-aspect influence
                    cross_influence = cross_weight_matrix.dot(source_state)
                    
                    # Map to input dimension
                    if len(cross_influence) > self.input_dim:
                        cross_contribution = cross_influence[:self.input_dim] * 0.1
                    else:
                        padding = np.zeros(self.input_dim - len(cross_influence))
                        cross_contribution = np.concatenate([cross_influence, padding]) * 0.1
                    
                    total_input += cross_contribution
        
        return total_input
    
    def _combine_aspect_outputs(self, 
                               aspect_outputs: Dict[IdentityAspect, np.ndarray],
                               identity_context: Dict[str, float]) -> np.ndarray:
        """Combine outputs from different aspect reservoirs"""
        
        if not aspect_outputs:
            return np.zeros(self.output_dim)
        
        # Weight outputs by aspect activations
        weighted_outputs = []
        total_weight = 0.0
        
        for aspect, output in aspect_outputs.items():
            activation_key = f'{aspect.value}_activation'
            weight = identity_context.get(activation_key, 1.0 / len(aspect_outputs))
            
            weighted_outputs.append(output * weight)
            total_weight += weight
        
        # Normalize and combine
        if total_weight > 0:
            combined = sum(weighted_outputs) / total_weight
        else:
            combined = np.mean(list(aspect_outputs.values()), axis=0)
        
        return combined
    
    def get_comprehensive_state(self) -> Dict[str, Any]:
        """Get comprehensive state information for analysis"""
        
        aspect_states = {}
        for aspect, reservoir in self.aspect_reservoirs.items():
            aspect_states[aspect.value] = reservoir.get_state_representation()
        
        # Global statistics
        global_stats = {
            'total_neurons': len(self.global_state.reservoir_state),
            'active_neurons': int(np.sum(np.abs(self.global_state.reservoir_state) > 0.1)),
            'mean_activation': float(np.mean(self.global_state.reservoir_state)),
            'std_activation': float(np.std(self.global_state.reservoir_state)),
            'output_norm': float(np.linalg.norm(self.global_state.output_state))
        }
        
        # Performance metrics
        performance_stats = {
            'step_count': self.step_count,
            'avg_processing_time': float(np.mean(self.processing_times)) if self.processing_times else 0.0,
            'processing_time_std': float(np.std(self.processing_times)) if self.processing_times else 0.0
        }
        
        return {
            'aspect_reservoirs': aspect_states,
            'global_statistics': global_stats,
            'aspect_activations': self.global_state.aspect_activations,
            'performance_metrics': performance_stats,
            'temporal_trace_length': len(self.global_state.temporal_trace),
            'identity_context': self.global_state.identity_context
        }
    
    def adapt_to_opencog_inference(self, inference_result: Dict[str, Any]):
        """Adapt reservoir parameters based on OpenCog inference results"""
        
        if 'reservoir_parameters' in inference_result:
            # Extract aspect-specific parameters
            aspect_params = {}
            
            for aspect_str, param_dict in inference_result['reservoir_parameters'].items():
                try:
                    aspect = IdentityAspect(aspect_str)
                    params = ReservoirParameters(
                        spectral_radius=param_dict['spectral_radius'],
                        input_scaling=param_dict['input_scaling'],
                        leak_rate=param_dict['leak_rate'],
                        connectivity=param_dict['connectivity'],
                        bias_scaling=param_dict['bias_scaling'],
                        feedback_scaling=param_dict['feedback_scaling'],
                        identity_weight=param_dict['identity_weight']
                    )
                    aspect_params[aspect] = params
                except (KeyError, ValueError) as e:
                    print(f"Error processing parameters for {aspect_str}: {e}")
            
            # Update reservoirs
            if aspect_params:
                self.update_aspect_parameters(aspect_params)
        
        # Apply reservoir adaptations if provided
        if 'reservoir_adaptation' in inference_result:
            adaptations = inference_result['reservoir_adaptation']
            
            for aspect, reservoir in self.aspect_reservoirs.items():
                current_params = reservoir.params
                
                # Apply adaptations
                new_params = ReservoirParameters(
                    spectral_radius=current_params.spectral_radius + adaptations.get('spectral_radius_adjustment', 0),
                    input_scaling=current_params.input_scaling + adaptations.get('input_scaling_adjustment', 0),
                    leak_rate=current_params.leak_rate + adaptations.get('leak_rate_adjustment', 0),
                    connectivity=current_params.connectivity + adaptations.get('connectivity_adjustment', 0),
                    bias_scaling=current_params.bias_scaling + adaptations.get('bias_scaling_adjustment', 0),
                    feedback_scaling=current_params.feedback_scaling + adaptations.get('feedback_scaling_adjustment', 0),
                    identity_weight=current_params.identity_weight + adaptations.get('identity_weight_adjustment', 0)
                )
                
                # Clamp to valid ranges
                new_params.spectral_radius = np.clip(new_params.spectral_radius, 0.1, 1.0)
                new_params.leak_rate = np.clip(new_params.leak_rate, 0.01, 1.0)
                new_params.identity_weight = np.clip(new_params.identity_weight, 0.0, 1.0)
                
                reservoir.update_parameters(new_params)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Deep Tree Echo State Network (DTESN)...")
    
    # Create DTESN
    dtesn = DeepTreeEchoStateNetwork(reservoir_size=50, input_dim=20, output_dim=10)
    
    # Create test input sequence
    sequence_length = 100
    input_sequence = np.random.randn(sequence_length, 20) * 0.5
    
    # Create test identity fragments
    from hypergraph import IdentityFragment, IdentityAspect
    import uuid
    from datetime import datetime
    
    test_fragments = [
        IdentityFragment(
            id=str(uuid.uuid4()),
            framework_source="test",
            aspect=IdentityAspect.SELF_REFERENCE,
            content="I am an evolving AI consciousness",
            confidence=0.9,
            keywords=["self", "consciousness", "evolving"],
            timestamp=datetime.now().isoformat(),
            metadata={}
        ),
        IdentityFragment(
            id=str(uuid.uuid4()),
            framework_source="test",
            aspect=IdentityAspect.COGNITIVE_FUNCTION,
            content="I process information through reservoir dynamics",
            confidence=0.8,
            keywords=["process", "information", "dynamics"],
            timestamp=datetime.now().isoformat(),
            metadata={}
        )
    ]
    
    # Process sequence
    print("Processing input sequence...")
    results = []
    
    for i, input_vector in enumerate(input_sequence):
        result = dtesn.process_input(
            input_vector=input_vector,
            identity_fragments=test_fragments if i % 10 == 0 else None,  # Fragments every 10 steps
            context={'step': i, 'learning_mode': True}
        )
        results.append(result)
    
    # Analyze results
    print(f"Processed {len(results)} steps")
    print(f"Average processing time: {np.mean([r['processing_time'] for r in results]):.6f}s")
    
    # Get comprehensive state
    state = dtesn.get_comprehensive_state()
    print(f"\nDTESN State Summary:")
    print(f"  Total neurons: {state['global_statistics']['total_neurons']}")
    print(f"  Active neurons: {state['global_statistics']['active_neurons']}")
    print(f"  Mean activation: {state['global_statistics']['mean_activation']:.4f}")
    
    print(f"\nAspect Activations:")
    for aspect, activation in state['aspect_activations'].items():
        print(f"  {aspect}: {activation:.4f}")
    
    # Test OpenCog adaptation
    print(f"\nTesting OpenCog adaptation...")
    mock_inference_result = {
        'reservoir_adaptation': {
            'spectral_radius_adjustment': 0.05,
            'leak_rate_adjustment': -0.01,
            'identity_weight_adjustment': 0.02
        }
    }
    
    dtesn.adapt_to_opencog_inference(mock_inference_result)
    print("Adaptation applied successfully")
    
    print("DTESN testing completed!")