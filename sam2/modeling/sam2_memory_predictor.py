from typing import Dict, Optional, Any
import torch

from .sam2_base import SAM2Base
from .memory_strategies.base import BaseMemoryStrategy
from .memory_strategies.strategies.lru_quantized import LRUQuantizedMemory

class SAM2MemoryPredictor(SAM2Base):
    """
    SAM2 predictor with configurable memory strategies.
    """
    
    def __init__(
        self,
        *args,
        memory_strategy: str = "lru_quantized",
        memory_strategy_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize SAM2 predictor with memory strategy.
        
        Args:
            memory_strategy (str): Name of the memory strategy to use
            memory_strategy_kwargs (Dict[str, Any]): Arguments for the memory strategy
            *args, **kwargs: Arguments for SAM2Base
        """
        super().__init__(*args, **kwargs)
        
        # Initialize memory strategy
        self.memory_strategy = self._create_memory_strategy(
            memory_strategy,
            memory_strategy_kwargs or {}
        )
    
    def _create_memory_strategy(
        self,
        strategy_name: str,
        kwargs: Dict[str, Any]
    ) -> BaseMemoryStrategy:
        """
        Create memory strategy instance.
        
        Args:
            strategy_name (str): Name of the strategy to create
            kwargs (Dict[str, Any]): Arguments for the strategy
            
        Returns:
            BaseMemoryStrategy: Created memory strategy instance
            
        Raises:
            ValueError: If unknown strategy name is provided
        """
        if strategy_name == "lru_quantized":
            return LRUQuantizedMemory(**kwargs)
        else:
            raise ValueError(f"Unknown memory strategy: {strategy_name}")
    
    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
        frame_idx: int = 0,
    ):
        """
        Encode new memory and add it to the memory strategy.
        
        Args:
            current_vision_feats: Current frame's visual features
            feat_sizes: Feature sizes
            pred_masks_high_res: High resolution predicted masks
            object_score_logits: Object score logits
            is_mask_from_pts: Whether masks are from points
            frame_idx (int): Index of the current frame
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Encoded memory features and positional encoding
        """
        # Get original memory encoding
        maskmem_features, maskmem_pos_enc = super()._encode_new_memory(
            current_vision_feats,
            feat_sizes,
            pred_masks_high_res,
            object_score_logits,
            is_mask_from_pts,
        )
        
        # Add to memory strategy
        memory = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc
        }
        self.memory_strategy.add_memory(frame_idx, memory)
        
        return maskmem_features, maskmem_pos_enc
    
    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        """
        Prepare memory conditioned features using the memory strategy.
        
        Args:
            frame_idx: Current frame index
            is_init_cond_frame: Whether this is an initial conditioning frame
            current_vision_feats: Current frame's visual features
            current_vision_pos_embeds: Current frame's positional embeddings
            feat_sizes: Feature sizes
            output_dict: Output dictionary
            num_frames: Number of frames
            track_in_reverse: Whether to track in reverse
            
        Returns:
            torch.Tensor: Memory conditioned features
        """
        if is_init_cond_frame:
            return super()._prepare_memory_conditioned_features(
                frame_idx,
                is_init_cond_frame,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                output_dict,
                num_frames,
                track_in_reverse,
            )
        
        # Get memories from strategy
        memories = self.memory_strategy.get_all_memories()
        
        # Prepare memory features
        memory_features = []
        memory_pos_embeds = []
        
        for mem_idx, memory in memories.items():
            if memory is not None:
                memory_features.append(memory["maskmem_features"])
                memory_pos_embeds.append(memory["maskmem_pos_enc"])
        
        if not memory_features:
            return super()._prepare_memory_conditioned_features(
                frame_idx,
                is_init_cond_frame,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                output_dict,
                num_frames,
                track_in_reverse,
            )
        
        # Concatenate memory features
        memory_features = torch.cat(memory_features, dim=0)
        memory_pos_embeds = torch.cat(memory_pos_embeds, dim=0)
        
        # Use memory attention
        return self.memory_attention(
            curr=current_vision_feats,
            memory=memory_features,
            curr_pos=current_vision_pos_embeds,
            memory_pos=memory_pos_embeds,
        ) 