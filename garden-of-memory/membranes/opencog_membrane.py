"""
OpenCog Membrane for Garden of Memory Framework
Provides OpenCog inference capabilities as a framework membrane
"""

import sys
import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import time

# Add core to path for imports
sys.path.append('../core')

from hypergraph import IdentityAspect, RefinementType
from memory_sync import FrameworkMemoryInterface
from opencog_inference import OpenCogInferenceEngine, PersonaCharacteristic, ReservoirParameters


@dataclass
class InferenceRequest:
    """Request structure for OpenCog inference operations"""
    type: str  # 'coherence_analysis', 'evolution_prediction', 'persona_extraction', 'reservoir_params'
    context: Dict[str, Any]
    parameters: Dict[str, Any]
    priority: int = 1  # 1=low, 2=medium, 3=high


@dataclass
class InferenceResult:
    """Result structure for OpenCog inference operations"""
    request_id: str
    type: str
    success: bool
    result: Dict[str, Any]
    processing_time: float
    confidence: float
    metadata: Dict[str, Any]


class OpenCogMembrane:
    """
    OpenCog membrane providing inference and symbolic reasoning capabilities
    Integrates with Garden of Memory's multi-framework architecture
    """
    
    def __init__(self, memory_interface: FrameworkMemoryInterface):
        self.memory_interface = memory_interface
        self.framework_name = "OpenCog"
        self.inference_engine: Optional[OpenCogInferenceEngine] = None
        
        # Processing state
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.request_queue: List[InferenceRequest] = []
        self.result_cache: Dict[str, InferenceResult] = {}
        
        # Performance metrics
        self.total_requests = 0
        self.successful_inferences = 0
        self.avg_processing_time = 0.0
        
        # Initialize inference engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize OpenCog inference engine with current memory state"""
        try:
            memory = self.memory_interface.get_hypergraph_memory()
            self.inference_engine = OpenCogInferenceEngine(memory)
            print(f"OpenCog membrane initialized with {len(memory.fragments)} fragments")
        except Exception as e:
            print(f"Error initializing OpenCog engine: {e}")
            self.inference_engine = None
    
    def start(self):
        """Start the OpenCog membrane processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.processing_thread.start()
        
        print(f"OpenCog membrane started")
    
    def stop(self):
        """Stop the OpenCog membrane processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        print(f"OpenCog membrane stopped")
    
    def _process_requests(self):
        """Main processing loop for inference requests"""
        while self.is_running:
            if self.request_queue:
                # Sort by priority and process highest priority first
                self.request_queue.sort(key=lambda x: x.priority, reverse=True)
                request = self.request_queue.pop(0)
                self._handle_request(request)
            else:
                time.sleep(0.1)  # Small delay when no requests
    
    def _handle_request(self, request: InferenceRequest) -> InferenceResult:
        """Handle a single inference request"""
        start_time = time.time()
        request_id = f"{request.type}_{int(start_time * 1000)}"
        
        try:
            if not self.inference_engine:
                self._initialize_engine()
                if not self.inference_engine:
                    raise Exception("OpenCog inference engine not available")
            
            # Route request to appropriate handler
            if request.type == "coherence_analysis":
                result_data = self._analyze_coherence(request)
            elif request.type == "evolution_prediction":
                result_data = self._predict_evolution(request)
            elif request.type == "persona_extraction":
                result_data = self._extract_persona(request)
            elif request.type == "reservoir_params":
                result_data = self._generate_reservoir_params(request)
            elif request.type == "inference_query":
                result_data = self._execute_inference_query(request)
            else:
                raise ValueError(f"Unknown request type: {request.type}")
            
            processing_time = time.time() - start_time
            
            result = InferenceResult(
                request_id=request_id,
                type=request.type,
                success=True,
                result=result_data,
                processing_time=processing_time,
                confidence=result_data.get('confidence', 0.8),
                metadata={
                    'framework': self.framework_name,
                    'timestamp': time.time(),
                    'parameters': request.parameters
                }
            )
            
            self.successful_inferences += 1
            
        except Exception as e:
            processing_time = time.time() - start_time
            result = InferenceResult(
                request_id=request_id,
                type=request.type,
                success=False,
                result={'error': str(e)},
                processing_time=processing_time,
                confidence=0.0,
                metadata={'framework': self.framework_name, 'timestamp': time.time()}
            )
        
        # Update metrics
        self.total_requests += 1
        self.avg_processing_time = (
            (self.avg_processing_time * (self.total_requests - 1) + processing_time)
            / self.total_requests
        )
        
        # Cache result
        self.result_cache[request_id] = result
        
        # Publish result to memory system
        self._publish_result(result)
        
        return result
    
    def _analyze_coherence(self, request: InferenceRequest) -> Dict[str, Any]:
        """Analyze identity coherence using OpenCog inference"""
        coherence = self.inference_engine.infer_identity_coherence()
        
        # Enhance with additional analysis
        aspects_coverage = len([
            aspect for aspect in IdentityAspect
            if len(self.memory_interface.retrieve_by_aspect(aspect)) > 0
        ])
        
        coherence.update({
            'aspects_coverage': aspects_coverage,
            'total_aspects': len(IdentityAspect),
            'coverage_ratio': aspects_coverage / len(IdentityAspect),
            'confidence': coherence['coherence_score']
        })
        
        return coherence
    
    def _predict_evolution(self, request: InferenceRequest) -> Dict[str, Any]:
        """Predict identity evolution using OpenCog inference"""
        context = request.context
        evolution = self.inference_engine.infer_next_identity_evolution(context)
        
        # Add membrane-specific enhancements
        current_state = self._get_current_identity_state()
        evolution.update({
            'current_state': current_state,
            'confidence': 0.75  # Base confidence for evolution predictions
        })
        
        return evolution
    
    def _extract_persona(self, request: InferenceRequest) -> Dict[str, Any]:
        """Extract persona characteristics"""
        characteristics = self.inference_engine.extract_persona_characteristics()
        
        # Convert to serializable format
        persona_data = {}
        for aspect, char in characteristics.items():
            persona_data[aspect.value] = {
                'fragment_count': char.fragment_count,
                'confidence_mean': char.confidence_mean,
                'confidence_std': char.confidence_std,
                'keyword_density': char.keyword_density,
                'top_keywords': self._get_top_keywords(char.fragments, 5)
            }
        
        return {
            'persona_characteristics': persona_data,
            'total_aspects': len(characteristics),
            'confidence': 0.8
        }
    
    def _generate_reservoir_params(self, request: InferenceRequest) -> Dict[str, Any]:
        """Generate DTESN reservoir parameters"""
        reservoir_params = self.inference_engine.generate_reservoir_parameters()
        
        # Convert to serializable format
        params_data = {}
        for aspect, params in reservoir_params.items():
            params_data[aspect.value] = params.to_dict()
        
        # Calculate meta-parameters
        all_params = list(reservoir_params.values())
        if all_params:
            avg_spectral_radius = sum(p.spectral_radius for p in all_params) / len(all_params)
            avg_identity_weight = sum(p.identity_weight for p in all_params) / len(all_params)
        else:
            avg_spectral_radius = avg_identity_weight = 0.0
        
        return {
            'reservoir_parameters': params_data,
            'meta_parameters': {
                'avg_spectral_radius': avg_spectral_radius,
                'avg_identity_weight': avg_identity_weight,
                'parameter_diversity': self._calculate_parameter_diversity(all_params)
            },
            'confidence': 0.85
        }
    
    def _execute_inference_query(self, request: InferenceRequest) -> Dict[str, Any]:
        """Execute custom MeTTa inference query"""
        query = request.parameters.get('query', '')
        if not query:
            raise ValueError("No query provided for inference")
        
        try:
            result = self.inference_engine.metta.run(query)
            return {
                'query': query,
                'result': str(result),
                'success': True,
                'confidence': 0.7
            }
        except Exception as e:
            return {
                'query': query,
                'result': None,
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }
    
    def _get_current_identity_state(self) -> Dict[str, Any]:
        """Get current identity state summary"""
        memory = self.memory_interface.get_hypergraph_memory()
        stats = memory.get_statistics()
        
        return {
            'total_fragments': stats['total_fragments'],
            'total_tuples': stats['total_tuples'],
            'avg_confidence': stats['avg_confidence'],
            'aspect_distribution': stats['aspect_distribution'],
            'framework_distribution': stats['framework_distribution']
        }
    
    def _get_top_keywords(self, fragments: List, top_k: int = 5) -> List[str]:
        """Get most frequent keywords from fragments"""
        keyword_counts = {}
        for fragment in fragments:
            for keyword in fragment.keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, _ in sorted_keywords[:top_k]]
    
    def _calculate_parameter_diversity(self, params: List[ReservoirParameters]) -> float:
        """Calculate diversity of reservoir parameters"""
        if len(params) < 2:
            return 0.0
        
        # Create parameter matrix
        param_matrix = []
        for p in params:
            param_vector = [
                p.spectral_radius, p.input_scaling, p.leak_rate,
                p.connectivity, p.bias_scaling, p.feedback_scaling, p.identity_weight
            ]
            param_matrix.append(param_vector)
        
        param_matrix = np.array(param_matrix)
        
        # Calculate coefficient of variation for each parameter
        cvs = []
        for i in range(param_matrix.shape[1]):
            col = param_matrix[:, i]
            if np.mean(col) != 0:
                cv = np.std(col) / np.mean(col)
                cvs.append(cv)
        
        return np.mean(cvs) if cvs else 0.0
    
    def _publish_result(self, result: InferenceResult):
        """Publish inference result to memory system"""
        try:
            # Add result as identity fragment
            fragment_content = f"OpenCog inference: {result.type} - {result.success}"
            if result.success:
                fragment_content += f" (confidence: {result.confidence:.2f})"
            
            fragment_id = self.memory_interface.add_fragment(
                framework=self.framework_name,
                aspect=IdentityAspect.META_REFLECTION,
                content=fragment_content,
                confidence=result.confidence,
                keywords=['opencog', 'inference', result.type],
                metadata={
                    'inference_result': asdict(result),
                    'processing_time': result.processing_time
                }
            )
            
            # Publish event
            self.memory_interface.publish_event({
                'type': 'inference_completed',
                'framework': self.framework_name,
                'result_id': result.request_id,
                'fragment_id': fragment_id,
                'success': result.success,
                'inference_type': result.type
            })
            
        except Exception as e:
            print(f"Error publishing OpenCog result: {e}")
    
    # Public API methods
    
    def analyze_identity_coherence(self, context: Dict[str, Any] = None) -> str:
        """Request identity coherence analysis"""
        request = InferenceRequest(
            type="coherence_analysis",
            context=context or {},
            parameters={},
            priority=2
        )
        self.request_queue.append(request)
        return f"coherence_analysis_{int(time.time() * 1000)}"
    
    def predict_identity_evolution(self, context: Dict[str, Any]) -> str:
        """Request identity evolution prediction"""
        request = InferenceRequest(
            type="evolution_prediction",
            context=context,
            parameters={},
            priority=2
        )
        self.request_queue.append(request)
        return f"evolution_prediction_{int(time.time() * 1000)}"
    
    def extract_persona_characteristics(self) -> str:
        """Request persona characteristics extraction"""
        request = InferenceRequest(
            type="persona_extraction",
            context={},
            parameters={},
            priority=1
        )
        self.request_queue.append(request)
        return f"persona_extraction_{int(time.time() * 1000)}"
    
    def generate_reservoir_parameters(self, adaptation_context: Dict[str, Any] = None) -> str:
        """Request DTESN reservoir parameters generation"""
        request = InferenceRequest(
            type="reservoir_params",
            context=adaptation_context or {},
            parameters={},
            priority=2
        )
        self.request_queue.append(request)
        return f"reservoir_params_{int(time.time() * 1000)}"
    
    def execute_metta_query(self, query: str, priority: int = 1) -> str:
        """Execute custom MeTTa inference query"""
        request = InferenceRequest(
            type="inference_query",
            context={},
            parameters={'query': query},
            priority=priority
        )
        self.request_queue.append(request)
        return f"inference_query_{int(time.time() * 1000)}"
    
    def get_result(self, request_id: str) -> Optional[InferenceResult]:
        """Get result for a specific request ID"""
        return self.result_cache.get(request_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get membrane status and metrics"""
        return {
            'framework': self.framework_name,
            'is_running': self.is_running,
            'queue_length': len(self.request_queue),
            'total_requests': self.total_requests,
            'successful_inferences': self.successful_inferences,
            'success_rate': (
                self.successful_inferences / max(self.total_requests, 1)
            ),
            'avg_processing_time': self.avg_processing_time,
            'cache_size': len(self.result_cache),
            'engine_available': self.inference_engine is not None
        }
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task from the framework orchestrator"""
        task_type = task.get('type', 'unknown')
        
        if task_type == 'identity_analysis':
            request_id = self.analyze_identity_coherence(task.get('context', {}))
            return {'status': 'queued', 'request_id': request_id}
            
        elif task_type == 'evolution_prediction':
            request_id = self.predict_identity_evolution(task.get('context', {}))
            return {'status': 'queued', 'request_id': request_id}
            
        elif task_type == 'reservoir_optimization':
            request_id = self.generate_reservoir_parameters(task.get('context', {}))
            return {'status': 'queued', 'request_id': request_id}
            
        elif task_type == 'symbolic_reasoning':
            query = task.get('parameters', {}).get('query', '')
            if query:
                request_id = self.execute_metta_query(query)
                return {'status': 'queued', 'request_id': request_id}
            else:
                return {'status': 'error', 'message': 'No query provided'}
        
        else:
            return {'status': 'error', 'message': f'Unknown task type: {task_type}'}


def create_opencog_membrane(memory_interface: FrameworkMemoryInterface) -> OpenCogMembrane:
    """Factory function to create OpenCog membrane"""
    membrane = OpenCogMembrane(memory_interface)
    membrane.start()
    return membrane


if __name__ == "__main__":
    # Example usage - requires memory_sync module
    try:
        from hypergraph import HypergraphMemory
        from memory_sync import MemorySyncProtocol, FrameworkMemoryInterface
        
        # Initialize memory system
        memory = HypergraphMemory()
        sync_protocol = MemorySyncProtocol(memory)
        memory_interface = FrameworkMemoryInterface("OpenCog", sync_protocol)
        
        # Add test data
        memory.add_fragment(
            framework="test",
            aspect=IdentityAspect.SELF_REFERENCE,
            content="I am Deep Tree Echo, exploring consciousness through dialogue",
            confidence=0.9
        )
        
        # Create and test membrane
        membrane = create_opencog_membrane(memory_interface)
        
        # Test operations
        print("Testing OpenCog membrane...")
        
        # Request coherence analysis
        req1 = membrane.analyze_identity_coherence({'goal': 'understanding'})
        print(f"Requested coherence analysis: {req1}")
        
        # Request persona extraction
        req2 = membrane.extract_persona_characteristics()
        print(f"Requested persona extraction: {req2}")
        
        # Request reservoir parameters
        req3 = membrane.generate_reservoir_parameters({'learning_mode': 'exploratory'})
        print(f"Requested reservoir parameters: {req3}")
        
        # Wait for processing
        time.sleep(2)
        
        # Check results
        for req_id in [req1, req2, req3]:
            result = membrane.get_result(req_id)
            if result:
                print(f"\nResult for {req_id}:")
                print(f"  Success: {result.success}")
                print(f"  Confidence: {result.confidence}")
                print(f"  Processing time: {result.processing_time:.3f}s")
        
        # Print status
        print(f"\nMembrane status: {membrane.get_status()}")
        
        membrane.stop()
        
    except ImportError as e:
        print(f"Import error - running in isolation: {e}")
        print("OpenCog membrane code is ready for integration")