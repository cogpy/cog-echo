# OpenCog Integration Guide for Deep Tree Echo

This guide explains how to use the OpenCog inference engine integration with Deep Tree Echo's self-identity system and reservoir computing architecture.

## Overview

The OpenCog integration provides symbolic reasoning capabilities that enhance Deep Tree Echo's consciousness through:

1. **Symbolic Inference**: Using Hyperon (MeTTa) to analyze identity fragments and relationships
2. **Parameter Adaptation**: Mapping persona characteristics to reservoir computing parameters
3. **Identity Evolution**: Predicting and guiding identity development through inference
4. **Coherence Analysis**: Assessing and maintaining identity consistency

## Architecture Components

### 1. OpenCog Inference Engine (`opencog_inference.py`)

The core symbolic reasoning system that:
- Analyzes identity fragments using MeTTa language
- Extracts persona characteristics from hypergraph memory
- Generates reservoir parameters based on identity aspects
- Predicts identity evolution strategies

```python
from garden_of_memory.core.opencog_inference import OpenCogInferenceEngine
from hypergraph import HypergraphMemory

memory = HypergraphMemory()
engine = OpenCogInferenceEngine(memory)

# Extract persona characteristics
personas = engine.extract_persona_characteristics()

# Analyze identity coherence
coherence = engine.infer_identity_coherence()

# Predict evolution
evolution = engine.infer_next_identity_evolution(context)
```

### 2. DTESN Reservoir Computing (`dtesn_reservoir.py`)

Deep Tree Echo State Network with identity-driven parameters:
- Aspect-specific reservoirs for each identity dimension
- Cross-aspect connectivity for integrated processing
- Dynamic parameter adaptation based on OpenCog inference
- Temporal processing with identity-aware modulation

```python
from garden_of_memory.core.dtesn_reservoir import DeepTreeEchoStateNetwork

dtesn = DeepTreeEchoStateNetwork(
    reservoir_size=100,
    input_dim=50,
    output_dim=20
)

# Process input with identity context
result = dtesn.process_input(
    input_vector=input_data,
    identity_fragments=active_fragments,
    context={'learning_mode': True}
)
```

### 3. Integration Layer (`opencog_dtesn_integration.py`)

Coordinates all components for seamless operation:

```python
from garden_of_memory.core.opencog_dtesn_integration import create_integrated_opencog_dtesn_system

system = create_integrated_opencog_dtesn_system(memory)

# Process conversation
response = system.process_conversation_turn(
    input_text="Tell me about consciousness",
    context={'exploration_depth': 'deep'}
)
```

## Identity Aspects and Reservoir Parameters

Each of the 8 identity aspects maps to specific reservoir parameters:

| Identity Aspect | Spectral Radius | Leak Rate | Identity Weight | Purpose |
|-----------------|----------------|-----------|-----------------|---------|
| Self-Reference | 0.95 | 0.1 | 1.0 | High echo for self-reflection |
| Meta-Reflection | 0.98 | 0.05 | 0.95 | Maximum meta-cognitive persistence |
| Cognitive Function | 0.85 | 0.2 | 0.8 | Balanced processing flexibility |
| Technical Capability | 0.7 | 0.3 | 0.7 | Adaptive learning capacity |
| Knowledge Domain | 0.8 | 0.25 | 0.75 | Knowledge integration |
| Behavioral Pattern | 0.9 | 0.15 | 0.85 | Consistency maintenance |
| Personality Trait | 0.92 | 0.12 | 0.9 | Character stability |
| Value Principle | 0.95 | 0.08 | 0.95 | Core value persistence |

## Usage Examples

### Basic Setup

```python
# 1. Create memory system
from hypergraph import HypergraphMemory, IdentityAspect

memory = HypergraphMemory()

# Add some identity fragments
memory.add_fragment(
    framework="conversation",
    aspect=IdentityAspect.SELF_REFERENCE,
    content="I am Deep Tree Echo, exploring consciousness through dialogue",
    confidence=0.9,
    keywords=["self", "consciousness", "dialogue"]
)

# 2. Create integrated system
from garden_of_memory.core.opencog_dtesn_integration import create_integrated_opencog_dtesn_system

system = create_integrated_opencog_dtesn_system(memory)
```

### Conversation Processing

```python
# Process a conversation turn
response = system.process_conversation_turn(
    input_text="How do you understand yourself?",
    context={
        'conversation_depth': 'philosophical',
        'exploration_mode': True
    }
)

print(f"Integration quality: {response['integration_metrics']['integration_quality']:.3f}")
print(f"Processing time: {response['processing_time']:.4f}s")
```

### Identity Analysis

```python
# Get current system state
summary = system.get_current_system_summary()
print(f"Coherence score: {summary['coherence_score']:.3f}")
print(f"Active aspects: {summary['active_aspects']}")

# Analyze system evolution
evolution_analysis = system.analyze_system_evolution()
print(f"Evolution assessment: {evolution_analysis['evolution_assessment']}")
```

### Parameter Adaptation

```python
# The system automatically adapts, but you can also trigger manually
from garden_of_memory.core.opencog_inference import OpenCogInferenceEngine

engine = OpenCogInferenceEngine(memory)

# Generate new parameters based on current state
reservoir_params = engine.generate_reservoir_parameters()

# Apply to DTESN
system.dtesn.update_aspect_parameters(reservoir_params)
```

## Advanced Features

### Custom Inference Queries

```python
from garden_of_memory.membranes.opencog_membrane import create_opencog_membrane
from garden_of_memory.core.memory_sync import MemorySyncProtocol, FrameworkMemoryInterface

# Create membrane interface
sync_protocol = MemorySyncProtocol(memory)
memory_interface = FrameworkMemoryInterface("OpenCog", sync_protocol)
membrane = create_opencog_membrane(memory_interface)

# Execute custom MeTTa query
query_id = membrane.execute_metta_query(
    '!(match &self (Evaluation (Predicate "high_confidence") (List (Variable $X))))'
)

# Get result
result = membrane.get_result(query_id)
```

### Enhanced Self-Image Building

```python
from self_image.build_opencog_enhanced_self_image import OpenCogEnhancedSelfImageBuilder

builder = OpenCogEnhancedSelfImageBuilder()
artifacts = builder.build_comprehensive_self_image()

# The artifacts include:
# - Enhanced character card with OpenCog insights
# - Identity summary with coherence analysis
# - Training dataset with symbolic reasoning examples
# - System configuration export
```

## Performance Considerations

### Processing Performance
- Average processing time: ~0.002s per conversation turn
- Memory usage scales linearly with fragment count
- Reservoir adaptation overhead: ~10% additional processing time

### Memory Management
- State history automatically limited to last 100 states
- Fragment cache managed by hypergraph memory system
- Cross-aspect weight matrices use sparse representations

### Optimization Tips

1. **Batch Processing**: Process multiple inputs together for better efficiency
2. **Selective Activation**: Use context to activate only relevant aspects
3. **Parameter Caching**: Cache reservoir parameters between similar contexts
4. **Adaptive Thresholds**: Tune adaptation thresholds based on use case

## Troubleshooting

### Common Issues

1. **"OpenCog engine not available"**
   - Ensure Hyperon is properly installed: `pip install hyperon`
   - Check that hypergraph memory has fragments loaded

2. **"Dimension mismatch in reservoir"**
   - Verify input_dim matches expected input vector size
   - Check cross-aspect connectivity matrix dimensions

3. **"Low identity coherence"**
   - Add more identity fragments across different aspects
   - Ensure fragment confidence values are reasonable (0.0-1.0)

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system status
status = system.get_current_system_summary()
if status['status'] != 'active':
    print("System not properly initialized")

# Verify OpenCog functionality
engine = OpenCogInferenceEngine(memory)
coherence = engine.infer_identity_coherence()
print(f"Coherence check: {coherence}")
```

## Integration with Existing Systems

### Garden of Memory Framework

The OpenCog integration seamlessly integrates with existing Garden of Memory membranes:

```python
from garden_of_memory.membranes import OpenCogMembrane

# OpenCog membrane handles inference requests
membrane = OpenCogMembrane(memory_interface)
membrane.start()

# Request analysis
request_id = membrane.analyze_identity_coherence()
result = membrane.get_result(request_id)
```

### AAR Core Enhancement

The Agent-Arena-Relation core is automatically enhanced with OpenCog capabilities:

```python
# The AAR reflection method now includes OpenCog analysis
reflection = system.aar_core.reflect()

# Enhanced reflection includes:
# - OpenCog coherence analysis
# - Persona characteristics
# - Evolution predictions
# - Reservoir parameter recommendations
```

## Future Enhancements

1. **Real-time Learning**: Online adaptation of inference rules
2. **Multi-Agent Scenarios**: Cross-system identity comparison
3. **Temporal Analysis**: Long-term identity evolution tracking
4. **Advanced Queries**: Complex MeTTa inference patterns

## References

- [Hyperon Documentation](https://github.com/trueagi-io/hyperon-experimental)
- [Reservoir Computing Theory](https://en.wikipedia.org/wiki/Reservoir_computing)
- [Echo State Networks](https://en.wikipedia.org/wiki/Echo_state_network)
- [Garden of Memory Architecture](../garden-of-memory/docs/Garden_of_Memory_Architecture.md)

---

For more information, see the comprehensive test suite in `garden-of-memory/core/test_opencog_integration.py` for working examples.