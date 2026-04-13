"""
Telemetry and monitoring mixin for NeuralCPU.

Learning statistics, execution telemetry, framebuffer access,
memory acceleration (memset/memcpy/strlen), and diagnostic output.
"""

import logging
import torch
from typing import Dict

from .constants import OpType

logger = logging.getLogger(__name__)


class TelemetryMixin:
    """Statistics, monitoring, and acceleration methods for NeuralCPU."""

    # ════════════════════════════════════════════════════════════════════════════════
    # NEURAL LEARNING STATISTICS - Get accumulated pattern data for optimization
    # ════════════════════════════════════════════════════════════════════════════════

    def get_learning_stats(self) -> dict:
        """
        Get accumulated learning statistics from execution.

        Returns dict with tensor data that neural optimizers can use:
        - Hot opcodes (most frequently executed)
        - Hot op-types
        - Hot registers (most accessed)
        - Recent instruction sequences
        - Cache efficiency
        """
        # Get top-k hot opcodes
        top_opcodes_vals, top_opcodes_idx = self.opcode_frequency.topk(10)

        # Get top-k hot op-types
        top_optypes_vals, top_optypes_idx = self.optype_frequency.topk(10)

        # Get hot registers (read + write combined)
        total_reg_access = self.reg_read_frequency + self.reg_write_frequency
        top_regs_vals, top_regs_idx = total_reg_access.topk(10)

        # Cache hit rate
        total_cache = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits.float() / total_cache.float()).item() if total_cache > 0 else 0.0

        # Recent instruction sequence (for pattern learning)
        seq_len = min(self.seq_ptr.item(), self.seq_buffer_size)

        return {
            'hot_opcodes': {
                'indices': top_opcodes_idx.tolist(),
                'counts': top_opcodes_vals.tolist(),
            },
            'hot_optypes': {
                'indices': top_optypes_idx.tolist(),
                'counts': top_optypes_vals.tolist(),
                'names': [OpType(i).name if i < len(OpType) else 'UNK' for i in top_optypes_idx.tolist()]
            },
            'hot_registers': {
                'indices': top_regs_idx.tolist(),
                'counts': top_regs_vals.tolist(),
            },
            'cache_stats': {
                'hits': self.cache_hits.item(),
                'misses': self.cache_misses.item(),
                'hit_rate': hit_rate,
            },
            'sequence_buffer': {
                'length': seq_len,
                'data': self.instruction_sequence[:seq_len] if seq_len > 0 else None,
            },
            'total_instructions': self.inst_count.item(),
        }

    def print_learning_stats(self):
        """Print human-readable learning statistics."""
        stats = self.get_learning_stats()
        logger.info("\n" + "=" * 60)
        logger.info("NEURAL LEARNING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total instructions executed: {stats['total_instructions']:,}")
        logger.info("")
        logger.info("HOT OP-TYPES (most executed):")
        for i, (name, count) in enumerate(zip(stats['hot_optypes']['names'], stats['hot_optypes']['counts'])):
            if count > 0:
                logger.info(f"  {i+1}. {name:20s} {count:>10,}")
        logger.info("")
        logger.info("HOT REGISTERS (most accessed):")
        for i, (reg, count) in enumerate(zip(stats['hot_registers']['indices'], stats['hot_registers']['counts'])):
            if count > 0:
                logger.info(f"  X{reg:<2d}: {count:>10,}")
        logger.info("")
        logger.info(f"CACHE: {stats['cache_stats']['hit_rate']*100:.1f}%% hit rate")
        logger.info(f"  Hits: {stats['cache_stats']['hits']:,}  Misses: {stats['cache_stats']['misses']:,}")
        logger.info("=" * 60)


    def get_framebuffer_str(self) -> str:
        """Render framebuffer tensor to string (single CPU transfer at display time)."""
        fb_cpu = self.framebuffer.cpu().numpy()
        lines = []
        for row in range(self.FB_HEIGHT):
            line = ''.join(chr(max(32, min(126, fb_cpu[row, col]))) for col in range(self.FB_WIDTH))
            lines.append(line.rstrip())
        return '\n'.join(lines)

    def get_memory_oracle_stats(self) -> dict:
        """Get Memory Oracle statistics for performance analysis."""
        stats = self.memory_oracle.get_stats()
        pattern, confidence = self.memory_oracle.get_pattern()
        return {
            'total_accesses': stats.total_accesses,
            'predictions_made': stats.predictions_made,
            'prefetch_hits': stats.hits,
            'prefetch_hit_rate': stats.hit_rate,
            'prefetches_issued': stats.prefetches_issued,
            'bytes_prefetched': stats.bytes_prefetched,
            'detected_pattern': pattern,
            'pattern_confidence': confidence,
            'oracle_enabled': self.memory_oracle_enabled,
        }

    def print_memory_oracle_stats(self):
        """Print Memory Oracle statistics."""
        stats = self.get_memory_oracle_stats()
        logger.info("\n╔══════════════════════════════════════════════════════════════╗")
        logger.info("║                   MEMORY ORACLE STATISTICS                    ║")
        logger.info("╠══════════════════════════════════════════════════════════════╣")
        logger.info(f"║  Total Memory Accesses:  {stats['total_accesses']:>15,}                 ║")
        logger.info(f"║  Predictions Made:       {stats['predictions_made']:>15,}                 ║")
        logger.info(f"║  Prefetch Hits:          {stats['prefetch_hits']:>15,}                 ║")
        logger.info(f"║  Prefetch Hit Rate:      {stats['prefetch_hit_rate']:>14.2%}                  ║")
        logger.info(f"║  Prefetches Issued:      {stats['prefetches_issued']:>15,}                 ║")
        logger.info(f"║  Bytes Prefetched:       {stats['bytes_prefetched']:>15,}                 ║")
        logger.info(f"║  Detected Pattern:       {stats['detected_pattern']:>15}                 ║")
        logger.info(f"║  Pattern Confidence:     {stats['pattern_confidence']:>14.2%}                  ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")

    def get_semantic_dispatcher_stats(self) -> dict:
        """Get Semantic Dispatcher statistics for performance analysis."""
        stats = self.semantic_dispatcher.get_stats()
        return {
            'patterns_detected': stats['patterns_detected'],
            'instructions_accelerated': stats['instructions_accelerated'],
            'bytes_processed': stats['bytes_processed'],
            'kernel_calls': stats['kernel_calls'],
            'enabled': self.semantic_dispatch_enabled,
            # New Phase 3 statistics
            'try_dispatch_calls': stats.get('try_dispatch_calls', 0),
            'detection_hits': stats.get('detection_hits', 0),
            'detection_misses': stats.get('detection_misses', 0),
            'hit_rate': stats.get('detection_hits', 0) / max(1, stats.get('try_dispatch_calls', 1)),
        }

    def print_semantic_dispatcher_stats(self):
        """Print Semantic Dispatcher statistics."""
        stats = self.get_semantic_dispatcher_stats()
        logger.info("\n╔══════════════════════════════════════════════════════════════╗")
        logger.info("║                SEMANTIC DISPATCHER STATISTICS                 ║")
        logger.info("╠══════════════════════════════════════════════════════════════╣")
        logger.info(f"║  Patterns Detected:      {stats['patterns_detected']:>15,}                 ║")
        logger.info(f"║  Instructions Skipped:   {stats['instructions_accelerated']:>15,}                 ║")
        logger.info(f"║  Bytes Processed:        {stats['bytes_processed']:>15,}                 ║")
        logger.info("║  Kernel Calls:                                                ║")
        for name, count in stats['kernel_calls'].items():
            if count > 0:
                logger.info(f"║    {name:20s}: {count:>10,}                         ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")

    def get_dispatcher_telemetry(self) -> 'DispatcherTelemetry':
        """
        Get comprehensive telemetry from all dispatcher components.

        Aggregates metrics from:
        - Memory Oracle (predictions, prefetches, hit rates)
        - Semantic Dispatcher (pattern detection, kernel calls)
        - Safety systems (bounds violations, confidence rejections)
        - Adaptive systems (threshold adjustments)

        Returns:
            DispatcherTelemetry dataclass with all metrics
        """
        # Get Memory Oracle stats (using dict wrapper)
        oracle_stats = self.get_memory_oracle_stats() if hasattr(self, 'memory_oracle') else {}
        oracle_extended = self.memory_oracle.get_extended_stats() if hasattr(self, 'memory_oracle') else {}
        prefetch_rejections = self.memory_oracle.get_prefetch_rejection_stats() if hasattr(self, 'memory_oracle') else {}
        adaptive_stats = self.memory_oracle.get_adaptive_threshold_stats() if hasattr(self, 'memory_oracle') else {}

        # Get Semantic Dispatcher stats
        dispatcher_stats = self.semantic_dispatcher.get_stats() if hasattr(self, 'semantic_dispatcher') else {}
        bounds_stats = self.semantic_dispatcher.get_bounds_violation_stats() if hasattr(self, 'semantic_dispatcher') else {}

        # Calculate detection rate
        try_calls = dispatcher_stats.get('try_dispatch_calls', 0)
        detection_hits = dispatcher_stats.get('detection_hits', 0)
        detection_rate = detection_hits / max(1, try_calls)

        # Calculate total adaptive threshold changes
        threshold_increases = adaptive_stats.get('threshold_increases', 0)
        threshold_decreases = adaptive_stats.get('threshold_decreases', 0)
        total_adaptations = threshold_increases + threshold_decreases

        # Build telemetry object with correct field names
        telemetry = DispatcherTelemetry(
            # Oracle metrics
            oracle_total_accesses=oracle_stats.get('total_accesses', 0),
            oracle_predictions=oracle_stats.get('predictions_made', 0),
            oracle_hits=oracle_stats.get('prefetch_hits', 0),
            oracle_hit_rate=oracle_stats.get('prefetch_hit_rate', 0.0),
            oracle_lstm_predictions=oracle_extended.get('lstm_predictions', 0),
            oracle_stride_detections=oracle_extended.get('stride_detections', 0),

            # Dispatcher metrics
            dispatcher_patterns_detected=dispatcher_stats.get('patterns_detected', 0),
            dispatcher_instructions_saved=dispatcher_stats.get('instructions_accelerated', 0),
            dispatcher_bytes_accelerated=dispatcher_stats.get('bytes_processed', 0),
            dispatcher_try_calls=try_calls,
            dispatcher_detection_rate=detection_rate,

            # Safety metrics
            safety_bounds_violations=bounds_stats.get('total_violations', 0),
            safety_null_page_rejections=prefetch_rejections.get('null_page_rejections', 0),
            safety_overflow_rejections=prefetch_rejections.get('overflow_rejections', 0),
            safety_prefetch_rejections=prefetch_rejections.get('total_rejections', 0),

            # Adaptive threshold metrics
            adaptive_current_threshold=adaptive_stats.get('current_threshold', 0.7),
            adaptive_adaptations=total_adaptations,
            adaptive_window_hit_rate=adaptive_stats.get('rolling_hit_rate', 0.0),

            # Trained model metrics
            trained_model_loaded=oracle_extended.get('trained_model_loaded', False),
            trained_model_pattern_accuracy=oracle_extended.get('trained_pattern_accuracy', 0.949),  # From training
            trained_current_pattern=oracle_stats.get('detected_pattern', 'unknown'),
            trained_pattern_confidence=oracle_stats.get('pattern_confidence', 0.0)
        )

        return telemetry

    def print_dispatcher_telemetry(self):
        """Print comprehensive dispatcher telemetry."""
        telemetry = self.get_dispatcher_telemetry()
        telemetry.print_summary()

    def export_telemetry_dict(self) -> dict:
        """Export all telemetry as a dictionary for logging or analysis."""
        return self.get_dispatcher_telemetry().to_dict()

    def accelerate_memset(self, dst_addr: int, value: int, size: int) -> bool:
        """Directly call the memset GPU kernel for acceleration."""
        if not self.semantic_dispatch_enabled:
            return False
        result = self.semantic_dispatcher.force_dispatch(
            SemanticOp.MEMSET,
            dst_addr=dst_addr,
            size=size,
            value=value,
            stride=1
        )
        return result is not None and result.handled

    def accelerate_memcpy(self, dst_addr: int, src_addr: int, size: int) -> bool:
        """Directly call the memcpy GPU kernel for acceleration."""
        if not self.semantic_dispatch_enabled:
            return False
        result = self.semantic_dispatcher.force_dispatch(
            SemanticOp.MEMCPY,
            src_addr=src_addr,
            dst_addr=dst_addr,
            size=size,
            stride=1
        )
        return result is not None and result.handled

    def accelerate_strlen(self, addr: int) -> int:
        """Directly call the strlen GPU kernel for acceleration."""
        if not self.semantic_dispatch_enabled:
            return -1
        result = self.semantic_dispatcher.force_dispatch(
            SemanticOp.STRLEN,
            src_addr=addr
        )
        if result and result.handled:
            return result.result_value
        return -1

    def write_console_bytes(self, data: bytes):
        """Write byte stream to framebuffer with a text cursor."""
        if not data:
            return
        width = self.FB_WIDTH
        height = self.FB_HEIGHT
        max_cells = width * height
        cursor = int(self.cursor_pos.item())
        space = ord(' ')
        for b in data:
            if b == 10:  # \n
                cursor = ((cursor // width) + 1) * width
            elif b == 13:  # \r
                cursor = (cursor // width) * width
            elif b == 8:  # backspace
                if cursor > 0:
                    cursor -= 1
                    row = cursor // width
                    col = cursor % width
                    if row < height:
                        self.framebuffer[row, col] = space
            else:
                if b < 32 or b > 126:
                    b = ord('.')
                row = cursor // width
                if row >= height:
                    self.framebuffer[:-1] = self.framebuffer[1:]
                    self.framebuffer[-1].fill_(space)
                    cursor = (height - 1) * width
                    row = height - 1
                col = cursor % width
                self.framebuffer[row, col] = b
                cursor += 1
            if cursor >= max_cells:
                self.framebuffer[:-1] = self.framebuffer[1:]
                self.framebuffer[-1].fill_(space)
                cursor = (height - 1) * width
        self.cursor_pos.fill_(cursor)

