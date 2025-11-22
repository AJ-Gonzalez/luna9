"""
Parquet data loading and sampling utilities.

Efficiently loads large conversation datasets from Parquet files
using Polars for streaming and lazy evaluation.
"""

import polars as pl
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
import logging

logger = logging.getLogger(__name__)


def validate_schema(df: pl.DataFrame) -> bool:
    """
    Validate that DataFrame has required columns for conversation processing.

    Required columns: conversation_id, message_id, text, speaker
    Optional: timestamp, turn_index, metadata

    Returns:
        bool: True if schema is valid
    """
    required = {'conversation_id', 'text'}
    has_required = required.issubset(set(df.columns))

    if not has_required:
        missing = required - set(df.columns)
        logger.error(f"Missing required columns: {missing}")

    return has_required


def load_conversations(
    parquet_path: str,
    sample_size: Optional[int] = None,
    strategy: str = 'random',
    min_conversation_length: int = 4
) -> pl.DataFrame:
    """
    Load conversations from Parquet file with optional sampling.

    Uses Polars lazy evaluation to avoid loading entire file into memory.

    Args:
        parquet_path: Path to parquet file or directory
        sample_size: Number of conversations to sample (None = all)
        strategy: Sampling strategy - 'random', 'stratified' (by speaker), or 'systematic'
        min_conversation_length: Minimum messages per conversation

    Returns:
        DataFrame with sampled conversations
    """
    path = Path(parquet_path)

    if not path.exists():
        raise FileNotFoundError(f"Parquet path not found: {parquet_path}")

    # Load with lazy evaluation
    logger.info(f"Loading conversations from {parquet_path}")
    df = pl.scan_parquet(str(path))

    # Validate schema on small sample first
    sample_check = df.head(10).collect()
    if not validate_schema(sample_check):
        raise ValueError("Invalid parquet schema")

    # Filter by minimum conversation length
    df = (df
          .with_columns([
              pl.col('conversation_id').count().over('conversation_id').alias('conv_length')
          ])
          .filter(pl.col('conv_length') >= min_conversation_length))

    # Apply sampling strategy
    if sample_size is not None:
        if strategy == 'random':
            # Random sampling - collect first for LazyFrame
            df_collected = df.collect()
            total_convs = df_collected.select('conversation_id').n_unique()
            fraction = min(1.0, sample_size / total_convs)
            return df_collected.sample(fraction=fraction, seed=42)

        elif strategy == 'stratified':
            # Sample equally from each speaker (if speaker column exists)
            if 'speaker' in df.columns:
                df = (df
                     .group_by('speaker')
                     .map_groups(lambda group: group.sample(
                         n=min(sample_size // 3, group.height),
                         seed=42
                     )))
            else:
                logger.warning("Speaker column not found, falling back to random sampling")
                df = df.sample(fraction=min(1.0, sample_size / df.collect().height), seed=42)

        elif strategy == 'systematic':
            # Every nth conversation
            df_collected = df.collect()
            total_rows = df_collected.height
            step = max(1, total_rows // sample_size)
            df = df_collected[::step]
            return df

    return df.collect()


def sample_pairs(
    df: pl.DataFrame,
    pair_strategy: str = 'mixed',
    pairs_per_conversation: int = 5,
    max_pairs: Optional[int] = 5000
) -> List[Tuple[str, str, Dict]]:
    """
    Extract message pairs from conversations with metadata.

    Args:
        df: DataFrame with conversations
        pair_strategy: How to select pairs within conversations
            - 'sequential': Adjacent messages (Q&A style)
            - 'distant': Messages far apart (different topics)
            - 'random': Random pairs within conversation
            - 'mixed': Combination of all strategies
        pairs_per_conversation: How many pairs to extract per conversation
        max_pairs: Maximum total pairs to return

    Returns:
        List of (message1, message2, metadata) tuples
        metadata contains: conversation_id, distance (in turns), pair_type
    """
    pairs = []

    # Group by conversation
    conversations = df.group_by('conversation_id', maintain_order=True)
    n_conversations = 0

    for conv_id, group in conversations:
        n_conversations += 1
        # Sort by turn index if available, otherwise assume order
        if 'turn_index' in group.columns:
            group = group.sort('turn_index')

        messages = group['text'].to_list()
        n_messages = len(messages)

        if n_messages < 2:
            continue

        # Extract pairs based on strategy
        conv_pairs = []

        if pair_strategy in ['sequential', 'mixed']:
            # Adjacent pairs
            for i in range(min(n_messages - 1, pairs_per_conversation)):
                conv_pairs.append({
                    'msg1': messages[i],
                    'msg2': messages[i + 1],
                    'conversation_id': conv_id,
                    'distance': 1,
                    'pair_type': 'sequential'
                })

        if pair_strategy in ['distant', 'mixed'] and n_messages > 5:
            # Distant pairs (>= 3 turns apart)
            n_distant = pairs_per_conversation // 3
            for _ in range(n_distant):
                i = random.randint(0, n_messages - 4)
                j = random.randint(i + 3, n_messages - 1)
                conv_pairs.append({
                    'msg1': messages[i],
                    'msg2': messages[j],
                    'conversation_id': conv_id,
                    'distance': j - i,
                    'pair_type': 'distant'
                })

        if pair_strategy in ['random', 'mixed']:
            # Random pairs
            n_random = pairs_per_conversation // 3
            for _ in range(n_random):
                i, j = random.sample(range(n_messages), 2)
                if i > j:
                    i, j = j, i
                conv_pairs.append({
                    'msg1': messages[i],
                    'msg2': messages[j],
                    'conversation_id': conv_id,
                    'distance': j - i,
                    'pair_type': 'random'
                })

        pairs.extend(conv_pairs[:pairs_per_conversation])

        if max_pairs and len(pairs) >= max_pairs:
            break

    # Shuffle and cap
    random.shuffle(pairs)
    pairs = pairs[:max_pairs] if max_pairs else pairs

    logger.info(f"Extracted {len(pairs)} message pairs from {n_conversations} conversations")

    return pairs


def get_parquet_stats(filepath: str) -> Dict:
    """
    Get statistics about Parquet file without loading it fully.

    Returns:
        Dict with num_rows, num_columns, file_size_mb, columns
    """
    from pyarrow import parquet as pq

    path = Path(filepath)
    parquet_file = pq.ParquetFile(filepath)

    return {
        'num_rows': parquet_file.metadata.num_rows,
        'num_row_groups': parquet_file.metadata.num_row_groups,
        'num_columns': parquet_file.metadata.num_columns,
        'file_size_mb': path.stat().st_size / (1024 * 1024),
        'columns': parquet_file.schema_arrow.names,
    }


def validate_data_quality(df: pl.DataFrame) -> Dict[str, bool]:
    """
    Validate data quality before processing.

    Returns:
        Dict of quality checks and their pass/fail status
    """
    checks = {}

    # Required columns
    checks['has_required_columns'] = validate_schema(df)

    # No null texts
    checks['no_null_texts'] = df['text'].null_count() == 0

    # Reasonable conversation lengths
    conv_lengths = df.group_by('conversation_id').count()
    checks['min_conversation_length'] = conv_lengths['count'].min() >= 2
    checks['reasonable_max_length'] = conv_lengths['count'].max() < 500

    # Text length reasonable (not empty, not huge)
    if 'text' in df.columns:
        text_lengths = df.select(pl.col('text').str.len_chars())['text']
        checks['no_empty_messages'] = text_lengths.min() > 0
        checks['text_length_reasonable'] = text_lengths.max() < 5000

    # Log failures
    failed = [check for check, passed in checks.items() if not passed]
    if failed:
        logger.warning(f"Data quality checks failed: {failed}")

    return checks
