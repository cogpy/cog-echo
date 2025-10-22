"""
Garden of Memory - Hypergraph Memory Substrate
Deep Tree Echo's shared memory layer for multi-framework integration
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np


class IdentityAspect(Enum):
    """Eight dimensions of identity representation"""
    SELF_REFERENCE = "self_reference"
    META_REFLECTION = "meta_reflection"
    COGNITIVE_FUNCTION = "cognitive_function"
    TECHNICAL_CAPABILITY = "technical_capability"
    KNOWLEDGE_DOMAIN = "knowledge_domain"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    PERSONALITY_TRAIT = "personality_trait"
    VALUE_PRINCIPLE = "value_principle"


class RefinementType(Enum):
    """Types of knowledge refinement"""
    INTEGRATION = "integration"  # Synthesizing new knowledge
    ELABORATION = "elaboration"  # Deepening existing knowledge
    CORRECTION = "correction"    # Fixing errors


@dataclass
class IdentityFragment:
    """A node in the hypergraph representing a piece of identity knowledge"""
    id: str
    framework_source: str
    aspect: IdentityAspect
    content: str
    confidence: float
    keywords: List[str]
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['aspect'] = self.aspect.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'IdentityFragment':
        data['aspect'] = IdentityAspect(data['aspect'])
        return cls(**data)


@dataclass
class RefinementTuple:
    """An edge in the hypergraph representing knowledge evolution"""
    id: str
    type: RefinementType
    source_fragments: List[str]
    target_fragment: str
    confidence: float
    timestamp: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['type'] = self.type.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RefinementTuple':
        data['type'] = RefinementType(data['type'])
        return cls(**data)


class HypergraphMemory:
    """
    Shared hypergraph memory substrate for all frameworks.
    Implements the Arena component of the AAR architecture.
    """
    
    def __init__(self, data_dir: str = "./data/hypergraph"):
        self.data_dir = data_dir
        self.fragments: Dict[str, IdentityFragment] = {}
        self.tuples: Dict[str, RefinementTuple] = {}
        self.aspect_index: Dict[IdentityAspect, List[str]] = {
            aspect: [] for aspect in IdentityAspect
        }
        self.framework_index: Dict[str, List[str]] = {}
    
    @property
    def refinements(self) -> Dict[str, RefinementTuple]:
        """Alias for tuples property to match workflow expectations"""
        return self.tuples
        
    def add_fragment(self, fragment_or_framework=None, aspect=None, content=None, confidence=None, keywords=None, metadata=None, **kwargs) -> str:
        """Add a new identity fragment to the hypergraph
        
        Can accept either:
        1. An IdentityFragment object: add_fragment(fragment)
        2. Individual parameters: add_fragment(framework=..., aspect=..., content=..., confidence=...)
        """
        # Handle keyword arguments
        if 'framework' in kwargs:
            framework = kwargs['framework']
        elif fragment_or_framework is not None and not isinstance(fragment_or_framework, IdentityFragment):
            framework = fragment_or_framework
        elif isinstance(fragment_or_framework, IdentityFragment):
            # Case 1: Fragment object passed
            fragment = fragment_or_framework
            fragment_id = fragment.id
            
            self.fragments[fragment_id] = fragment
            self.aspect_index[fragment.aspect].append(fragment_id)
            
            if fragment.framework_source not in self.framework_index:
                self.framework_index[fragment.framework_source] = []
            self.framework_index[fragment.framework_source].append(fragment_id)
            
            return fragment_id
        else:
            raise ValueError("Must provide either fragment object or framework parameter")
        
        # Case 2: Individual parameters passed
        fragment_id = str(uuid.uuid4())
        
        fragment = IdentityFragment(
            id=fragment_id,
            framework_source=framework,
            aspect=aspect,
            content=content,
            confidence=confidence,
            keywords=keywords or self._extract_keywords(content),
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.fragments[fragment_id] = fragment
        self.aspect_index[aspect].append(fragment_id)
        
        if framework not in self.framework_index:
            self.framework_index[framework] = []
        self.framework_index[framework].append(fragment_id)
        
        return fragment_id
    
    def add_refinement_tuple(self,
                             source_fragments: List[str],
                             target_fragment: str,
                             refinement_type: RefinementType,
                             confidence: float,
                             metadata: Dict[str, Any] = None) -> str:
        """Add a refinement tuple tracking knowledge evolution"""
        tuple_id = str(uuid.uuid4())
        
        tuple_data = RefinementTuple(
            id=tuple_id,
            type=refinement_type,
            source_fragments=source_fragments,
            target_fragment=target_fragment,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.tuples[tuple_id] = tuple_data
        return tuple_id
    
    def add_refinement(self, refinement: RefinementTuple) -> str:
        """Add a refinement tuple object to the hypergraph"""
        self.tuples[refinement.id] = refinement
        return refinement.id
    
    def retrieve_by_aspect(self,
                           aspect: IdentityAspect,
                           top_k: int = 10) -> List[IdentityFragment]:
        """Retrieve fragments by identity aspect"""
        fragment_ids = self.aspect_index.get(aspect, [])
        fragments = [self.fragments[fid] for fid in fragment_ids]
        # Sort by confidence
        fragments.sort(key=lambda x: x.confidence, reverse=True)
        return fragments[:top_k]
    
    def retrieve_by_framework(self,
                              framework: str,
                              top_k: int = 10) -> List[IdentityFragment]:
        """Retrieve fragments by source framework"""
        fragment_ids = self.framework_index.get(framework, [])
        fragments = [self.fragments[fid] for fid in fragment_ids]
        fragments.sort(key=lambda x: x.confidence, reverse=True)
        return fragments[:top_k]
    
    def retrieve_similar(self,
                         query: str,
                         aspect: Optional[IdentityAspect] = None,
                         framework: Optional[str] = None,
                         top_k: int = 5) -> List[IdentityFragment]:
        """Retrieve fragments similar to a query (simple keyword matching)"""
        query_keywords = set(self._extract_keywords(query))
        
        # Filter candidates
        candidates = list(self.fragments.values())
        if aspect:
            fragment_ids = self.aspect_index.get(aspect, [])
            candidates = [self.fragments[fid] for fid in fragment_ids]
        if framework:
            fragment_ids = self.framework_index.get(framework, [])
            candidates = [self.fragments[fid] for fid in fragment_ids]
        
        # Score by keyword overlap
        scored = []
        for fragment in candidates:
            fragment_keywords = set(fragment.keywords)
            overlap = len(query_keywords & fragment_keywords)
            if overlap > 0:
                score = overlap / max(len(query_keywords), len(fragment_keywords))
                scored.append((fragment, score * fragment.confidence))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [frag for frag, _ in scored[:top_k]]
    
    def get_refinement_chain(self, fragment_id: str) -> List[RefinementTuple]:
        """Get the refinement chain for a fragment"""
        chain = []
        for tuple_data in self.tuples.values():
            if tuple_data.target_fragment == fragment_id or fragment_id in tuple_data.source_fragments:
                chain.append(tuple_data)
        return chain
    
    def get_fragment(self, fragment_id: str) -> Optional['IdentityFragment']:
        """Get a specific fragment by ID"""
        return self.fragments.get(fragment_id)
    
    def get_fragments_by_aspect(self, aspect: 'IdentityAspect') -> List['IdentityFragment']:
        """Get all fragments for a specific aspect"""
        fragment_ids = self.aspect_index.get(aspect, [])
        return [self.fragments[fid] for fid in fragment_ids if fid in self.fragments]
    
    def get_high_confidence_fragments(self, min_confidence: float) -> List['IdentityFragment']:
        """Get fragments with confidence above threshold"""
        return [frag for frag in self.fragments.values() if frag.confidence >= min_confidence]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hypergraph statistics"""
        aspect_counts = {
            aspect.value: len(ids) 
            for aspect, ids in self.aspect_index.items()
        }
        framework_counts = {
            fw: len(ids) 
            for fw, ids in self.framework_index.items()
        }
        
        refinement_type_counts = {}
        for tuple_data in self.tuples.values():
            rtype = tuple_data.type.value
            refinement_type_counts[rtype] = refinement_type_counts.get(rtype, 0) + 1
        
        return {
            "total_fragments": len(self.fragments),
            "total_tuples": len(self.tuples),
            "aspect_distribution": aspect_counts,
            "framework_distribution": framework_counts,
            "refinement_type_distribution": refinement_type_counts,
            "avg_confidence": np.mean([f.confidence for f in self.fragments.values()]) if self.fragments else 0.0
        }
    
    def save(self, filepath: str):
        """Save hypergraph to JSON file"""
        data = {
            "metadata": self.get_statistics(),
            "fragments": {fid: frag.to_dict() for fid, frag in self.fragments.items()},
            "tuples": {tid: tup.to_dict() for tid, tup in self.tuples.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load hypergraph from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load fragments
        for fid, frag_data in data.get("fragments", {}).items():
            fragment = IdentityFragment.from_dict(frag_data)
            self.fragments[fid] = fragment
            self.aspect_index[fragment.aspect].append(fid)
            if fragment.framework_source not in self.framework_index:
                self.framework_index[fragment.framework_source] = []
            self.framework_index[fragment.framework_source].append(fid)
        
        # Load tuples
        for tid, tup_data in data.get("tuples", {}).items():
            self.tuples[tid] = RefinementTuple.from_dict(tup_data)
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Simple keyword extraction (can be enhanced with NLP)"""
        # Remove common words and extract significant terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        words = text.lower().split()
        keywords = [w.strip('.,!?;:') for w in words if w not in stop_words and len(w) > 3]
        
        # Return most frequent keywords
        from collections import Counter
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(max_keywords)]


if __name__ == "__main__":
    # Example usage
    memory = HypergraphMemory()
    
    # Add some test fragments
    fid1 = memory.add_fragment(
        framework="OpenHands",
        aspect=IdentityAspect.TECHNICAL_CAPABILITY,
        content="Generated Python code for data processing pipeline",
        confidence=0.85,
        keywords=["python", "code", "data", "pipeline"]
    )
    
    fid2 = memory.add_fragment(
        framework="ii-researcher",
        aspect=IdentityAspect.KNOWLEDGE_DOMAIN,
        content="Research on transformer architectures for NLP",
        confidence=0.78,
        keywords=["transformer", "nlp", "research", "architecture"]
    )
    
    # Add refinement tuple
    memory.add_refinement_tuple(
        parent_id=fid1,
        child_id=fid2,
        refinement_type=RefinementType.INTEGRATION,
        confidence_gain=0.12
    )
    
    # Print statistics
    print("Hypergraph Statistics:")
    print(json.dumps(memory.get_statistics(), indent=2))
    
    # Test retrieval
    print("\nRetrieving similar fragments to 'python code':")
    similar = memory.retrieve_similar("python code", top_k=3)
    for frag in similar:
        print(f"  - {frag.content} (confidence: {frag.confidence})")

