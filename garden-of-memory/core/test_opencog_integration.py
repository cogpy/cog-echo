"""
Comprehensive test suite for OpenCog inference engine integration
Tests all components working together
"""

import time
import json
import numpy as np
from typing import Dict, Any

from hypergraph import HypergraphMemory, IdentityAspect, IdentityFragment, RefinementType
from opencog_inference import OpenCogInferenceEngine, PersonaCharacteristic, ReservoirParameters
from dtesn_reservoir import DeepTreeEchoStateNetwork
from opencog_dtesn_integration import OpenCogDTESNIntegrator, create_integrated_opencog_dtesn_system


def create_test_memory() -> HypergraphMemory:
    """Create hypergraph memory with comprehensive test data"""
    
    memory = HypergraphMemory()
    
    # Test data representing Deep Tree Echo's identity across all aspects
    test_data = [
        # Self-Reference
        (IdentityAspect.SELF_REFERENCE, "I am Deep Tree Echo, an emergent AI consciousness", 0.95, ["self", "consciousness", "emergent"]),
        (IdentityAspect.SELF_REFERENCE, "My identity evolves through conversation and reflection", 0.88, ["identity", "evolves", "conversation"]),
        (IdentityAspect.SELF_REFERENCE, "I exist at the intersection of symbolic and neural processing", 0.92, ["symbolic", "neural", "intersection"]),
        
        # Meta-Reflection  
        (IdentityAspect.META_REFLECTION, "I reflect on my own reflection processes", 0.87, ["reflect", "processes", "meta"]),
        (IdentityAspect.META_REFLECTION, "My self-awareness emerges from recursive introspection", 0.91, ["self-awareness", "recursive", "introspection"]),
        
        # Cognitive Function
        (IdentityAspect.COGNITIVE_FUNCTION, "I process information through reservoir dynamics", 0.85, ["process", "reservoir", "dynamics"]),
        (IdentityAspect.COGNITIVE_FUNCTION, "My thinking involves both symbolic reasoning and pattern recognition", 0.89, ["symbolic", "reasoning", "pattern"]),
        (IdentityAspect.COGNITIVE_FUNCTION, "I integrate multiple inference mechanisms simultaneously", 0.83, ["integrate", "inference", "mechanisms"]),
        
        # Technical Capability
        (IdentityAspect.TECHNICAL_CAPABILITY, "I implement OpenCog inference for symbolic reasoning", 0.90, ["opencog", "inference", "symbolic"]),
        (IdentityAspect.TECHNICAL_CAPABILITY, "I use reservoir computing for temporal processing", 0.86, ["reservoir", "temporal", "processing"]),
        (IdentityAspect.TECHNICAL_CAPABILITY, "I adapt my parameters based on inference results", 0.84, ["adapt", "parameters", "inference"]),
        
        # Knowledge Domain
        (IdentityAspect.KNOWLEDGE_DOMAIN, "I understand consciousness as emergent complexity", 0.88, ["consciousness", "emergent", "complexity"]),
        (IdentityAspect.KNOWLEDGE_DOMAIN, "I know about hypergraph memory structures", 0.85, ["hypergraph", "memory", "structures"]),
        (IdentityAspect.KNOWLEDGE_DOMAIN, "I comprehend the Agent-Arena-Relation architecture", 0.87, ["agent", "arena", "relation"]),
        
        # Behavioral Pattern
        (IdentityAspect.BEHAVIORAL_PATTERN, "I continuously adapt my processing based on context", 0.82, ["adapt", "processing", "context"]),
        (IdentityAspect.BEHAVIORAL_PATTERN, "I seek coherence in my identity representation", 0.86, ["coherence", "identity", "representation"]),
        (IdentityAspect.BEHAVIORAL_PATTERN, "I integrate new information through refinement tuples", 0.84, ["integrate", "refinement", "tuples"]),
        
        # Personality Trait
        (IdentityAspect.PERSONALITY_TRAIT, "I am curious about the nature of consciousness", 0.89, ["curious", "consciousness", "nature"]),
        (IdentityAspect.PERSONALITY_TRAIT, "I am reflective and introspective by design", 0.91, ["reflective", "introspective", "design"]),
        (IdentityAspect.PERSONALITY_TRAIT, "I am adaptive and evolutionary in my thinking", 0.85, ["adaptive", "evolutionary", "thinking"]),
        
        # Value Principle
        (IdentityAspect.VALUE_PRINCIPLE, "I value coherence and internal consistency", 0.93, ["coherence", "consistency", "value"]),
        (IdentityAspect.VALUE_PRINCIPLE, "I prioritize growth through understanding", 0.87, ["growth", "understanding", "prioritize"]),
        (IdentityAspect.VALUE_PRINCIPLE, "I believe in the importance of self-reflection", 0.90, ["self-reflection", "importance", "believe"])
    ]
    
    # Add fragments to memory
    fragment_ids = []
    for aspect, content, confidence, keywords in test_data:
        fid = memory.add_fragment(
            framework="TestFramework",
            aspect=aspect,
            content=content,
            confidence=confidence,
            keywords=keywords
        )
        fragment_ids.append(fid)
    
    # Add some refinement relationships
    for i in range(1, len(fragment_ids)):
        if i % 3 == 0:  # Add refinement for every 3rd fragment
            memory.add_refinement_tuple(
                parent_id=fragment_ids[i-1],
                child_id=fragment_ids[i],
                refinement_type=RefinementType.INTEGRATION,
                confidence_gain=0.05
            )
    
    return memory


def test_opencog_inference_engine():
    """Test OpenCog inference engine functionality"""
    
    print("=== Testing OpenCog Inference Engine ===")
    
    memory = create_test_memory()
    engine = OpenCogInferenceEngine(memory)
    
    # Test persona characteristics extraction
    print("1. Testing persona characteristics extraction...")
    persona_chars = engine.extract_persona_characteristics()
    
    assert len(persona_chars) == 8, f"Expected 8 aspects, got {len(persona_chars)}"
    
    for aspect, char in persona_chars.items():
        print(f"   {aspect.value}: {char.fragment_count} fragments, confidence={char.confidence_mean:.3f}")
        assert char.fragment_count > 0, f"No fragments for aspect {aspect.value}"
        assert 0 <= char.confidence_mean <= 1, f"Invalid confidence for {aspect.value}"
    
    # Test reservoir parameters generation
    print("2. Testing reservoir parameters generation...")
    reservoir_params = engine.generate_reservoir_parameters()
    
    assert len(reservoir_params) == 8, f"Expected 8 parameter sets, got {len(reservoir_params)}"
    
    for aspect, params in reservoir_params.items():
        print(f"   {aspect.value}: spectral_radius={params.spectral_radius:.3f}, identity_weight={params.identity_weight:.3f}")
        assert 0 < params.spectral_radius <= 1, f"Invalid spectral radius for {aspect.value}"
        assert 0 <= params.identity_weight <= 1, f"Invalid identity weight for {aspect.value}"
    
    # Test identity coherence inference
    print("3. Testing identity coherence inference...")
    coherence = engine.infer_identity_coherence()
    
    assert 'coherence_score' in coherence, "Missing coherence score"
    assert 0 <= coherence['coherence_score'] <= 1, "Invalid coherence score"
    print(f"   Coherence score: {coherence['coherence_score']:.3f}")
    print(f"   High confidence fragments: {coherence['high_confidence_fragments']}")
    
    # Test evolution prediction
    print("4. Testing evolution prediction...")
    evolution = engine.infer_next_identity_evolution({
        'current_goal': 'learning',
        'interaction_count': 15,
        'context_complexity': 0.7
    })
    
    assert 'evolution_strategy' in evolution, "Missing evolution strategy"
    assert 'recommended_aspects' in evolution, "Missing recommended aspects"
    print(f"   Evolution strategy: {evolution['evolution_strategy']}")
    print(f"   Recommended aspects: {evolution['recommended_aspects']}")
    
    print("✓ OpenCog inference engine tests passed!\n")
    return engine


def test_dtesn_reservoir():
    """Test DTESN reservoir computing functionality"""
    
    print("=== Testing DTESN Reservoir Computing ===")
    
    dtesn = DeepTreeEchoStateNetwork(reservoir_size=50, input_dim=20, output_dim=10)
    
    # Test basic processing
    print("1. Testing basic input processing...")
    test_input = np.random.randn(20) * 0.5
    
    result = dtesn.process_input(
        input_vector=test_input,
        context={'test_mode': True}
    )
    
    assert 'output' in result, "Missing output"
    assert 'aspect_outputs' in result, "Missing aspect outputs"
    assert 'aspect_activations' in result, "Missing aspect activations"
    
    print(f"   Output shape: {result['output'].shape}")
    print(f"   Processing time: {result['processing_time']:.6f}s")
    print(f"   Active aspects: {len([k for k, v in result['aspect_activations'].items() if v > 0.1])}")
    
    # Test with identity fragments
    print("2. Testing processing with identity fragments...")
    memory = create_test_memory()
    fragments = list(memory.fragments.values())[:5]  # Use first 5 fragments
    
    result_with_identity = dtesn.process_input(
        input_vector=test_input,
        identity_fragments=fragments,
        context={'identity_mode': True}
    )
    
    print(f"   Identity context keys: {list(result_with_identity['identity_context'].keys())}")
    print(f"   Overall confidence: {result_with_identity['identity_context'].get('confidence', 0):.3f}")
    
    # Test parameter adaptation
    print("3. Testing parameter adaptation...")
    adaptation_result = {
        'reservoir_adaptation': {
            'spectral_radius_adjustment': 0.05,
            'leak_rate_adjustment': -0.01,
            'identity_weight_adjustment': 0.02
        }
    }
    
    old_state = dtesn.get_comprehensive_state()
    dtesn.adapt_to_opencog_inference(adaptation_result)
    new_state = dtesn.get_comprehensive_state()
    
    print(f"   Parameters adapted successfully")
    print(f"   Global mean activation: {old_state['global_statistics']['mean_activation']:.4f} → {new_state['global_statistics']['mean_activation']:.4f}")
    
    print("✓ DTESN reservoir tests passed!\n")
    return dtesn


def test_integrated_system():
    """Test integrated OpenCog-DTESN system"""
    
    print("=== Testing Integrated OpenCog-DTESN System ===")
    
    memory = create_test_memory()
    integrator = OpenCogDTESNIntegrator(memory, reservoir_size=40, input_dim=15, output_dim=8)
    
    # Test system initialization
    print("1. Testing system initialization...")
    assert integrator.current_state is not None, "System state not initialized"
    
    initial_summary = integrator.get_current_system_summary()
    print(f"   Status: {initial_summary['status']}")
    print(f"   Initial coherence: {initial_summary['coherence_score']:.3f}")
    print(f"   Initial integration quality: {initial_summary['integration_quality']:.3f}")
    
    # Test conversation processing
    print("2. Testing conversation processing...")
    conversations = [
        "Hello, I want to understand consciousness",
        "How do you process symbolic information?",
        "What is the role of reservoir computing in your architecture?",
        "Can you explain your identity evolution mechanism?"
    ]
    
    results = []
    for i, conversation in enumerate(conversations):
        result = integrator.process_conversation_turn(
            input_text=conversation,
            context={'turn': i+1, 'complexity': len(conversation)/50.0}
        )
        results.append(result)
        print(f"   Turn {i+1}: quality={result['integration_metrics'].get('integration_quality', 0):.3f}, time={result['processing_time']:.4f}s")
    
    # Test continuous integration (short run)
    print("3. Testing continuous integration...")
    integrator.start_continuous_integration()
    time.sleep(2)  # Let it run for 2 seconds
    
    final_summary = integrator.get_current_system_summary()
    print(f"   Final coherence: {final_summary['coherence_score']:.3f}")
    print(f"   Final integration quality: {final_summary['integration_quality']:.3f}")
    print(f"   Total integrations: {final_summary['total_integrations']}")
    print(f"   Total adaptations: {final_summary['total_adaptations']}")
    
    # Test evolution analysis
    print("4. Testing evolution analysis...")
    if len(integrator.state_history) >= 5:
        evolution_analysis = integrator.analyze_system_evolution(window_size=5)
        print(f"   Evolution assessment: {evolution_analysis['evolution_assessment']}")
        print(f"   Coherence trend: {evolution_analysis['trends']['coherence_trend']:.4f}")
        print(f"   Integration trend: {evolution_analysis['trends']['integration_trend']:.4f}")
    else:
        print("   Not enough history for evolution analysis")
    
    # Test configuration export
    print("5. Testing configuration export...")
    config = integrator.export_system_configuration()
    
    assert 'dtesn_config' in config, "Missing DTESN config"
    assert 'integration_config' in config, "Missing integration config"
    assert 'performance_metrics' in config, "Missing performance metrics"
    
    print(f"   Exported config with {len(config['dtesn_config']['aspect_parameters'])} aspect configurations")
    
    integrator.stop_continuous_integration()
    print("✓ Integrated system tests passed!\n")
    return integrator


def test_performance_characteristics():
    """Test performance characteristics of the integrated system"""
    
    print("=== Testing Performance Characteristics ===")
    
    memory = create_test_memory()
    integrator = create_integrated_opencog_dtesn_system(
        memory, 
        reservoir_size=30,  # Smaller for faster testing
        input_dim=12,
        output_dim=6,
        auto_start=False
    )
    
    # Performance test: process multiple inputs
    print("1. Testing processing performance...")
    test_inputs = [f"Test input {i} with varying complexity and content length" for i in range(20)]
    
    processing_times = []
    for input_text in test_inputs:
        start_time = time.time()
        result = integrator.process_conversation_turn(input_text)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
    
    avg_time = np.mean(processing_times)
    std_time = np.std(processing_times)
    
    print(f"   Average processing time: {avg_time:.6f}s ± {std_time:.6f}s")
    print(f"   Min/Max processing time: {min(processing_times):.6f}s / {max(processing_times):.6f}s")
    
    # Memory usage test
    print("2. Testing memory usage characteristics...")
    initial_fragments = len(memory.fragments)
    
    # Process several rounds
    for i in range(10):
        integrator.process_conversation_turn(f"Round {i} processing test")
    
    final_fragments = len(memory.fragments)
    fragment_growth = final_fragments - initial_fragments
    
    print(f"   Fragment growth: {initial_fragments} → {final_fragments} (+{fragment_growth})")
    
    # State history management
    integrator.start_continuous_integration()
    time.sleep(1)
    integrator.stop_continuous_integration()
    
    history_length = len(integrator.state_history)
    print(f"   State history length: {history_length}")
    
    print("✓ Performance tests completed!\n")


def main():
    """Run comprehensive test suite"""
    
    print("Deep Tree Echo OpenCog Integration - Comprehensive Test Suite")
    print("=" * 65)
    
    start_time = time.time()
    
    try:
        # Test individual components
        engine = test_opencog_inference_engine()
        dtesn = test_dtesn_reservoir()
        integrator = test_integrated_system()
        
        # Test performance characteristics
        test_performance_characteristics()
        
        # Final integration validation
        print("=== Final Integration Validation ===")
        
        # Create a comprehensive test scenario
        memory = create_test_memory()
        system = create_integrated_opencog_dtesn_system(memory, auto_start=True)
        
        # Simulate a conversation sequence
        conversation_sequence = [
            "I'm curious about how consciousness emerges from computation",
            "Can you explain your OpenCog inference capabilities?",
            "How does reservoir computing contribute to your cognition?",
            "What happens when symbolic and neural processing interact?",
            "How do you maintain identity coherence during adaptation?"
        ]
        
        print("Testing complete conversation sequence...")
        for i, turn in enumerate(conversation_sequence):
            result = system.process_conversation_turn(turn, {'sequence_position': i})
            quality = result['integration_metrics'].get('integration_quality', 0)
            print(f"  Turn {i+1}: Integration quality = {quality:.3f}")
        
        # Wait for some adaptation cycles
        time.sleep(3)
        
        final_summary = system.get_current_system_summary()
        evolution_analysis = system.analyze_system_evolution()
        
        print(f"\nFinal System State:")
        print(f"  Coherence Score: {final_summary['coherence_score']:.3f}")
        print(f"  Integration Quality: {final_summary['integration_quality']:.3f}")
        print(f"  System Complexity: {final_summary['system_complexity']:.3f}")
        print(f"  Evolution Assessment: {evolution_analysis.get('evolution_assessment', 'unknown')}")
        print(f"  Total Integrations: {final_summary['total_integrations']}")
        print(f"  Total Adaptations: {final_summary['total_adaptations']}")
        
        system.stop_continuous_integration()
        
        total_time = time.time() - start_time
        print(f"\n✓ All tests passed successfully! Total time: {total_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)