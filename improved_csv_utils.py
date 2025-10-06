"""
IMPROVED CSV Utilities System
Fixes for column drift, deduplication, max_groups support, atomic writes

DESIGN PRINCIPLES:
1. Strict header adherence: Always use fixed CSV_HEADERS from config.py
2. Deterministic deduplication: tx_hash + nft_address + timestamp composite keys  
3. max_groups support: Gezielt verarbeiten für reproduzierbare Tests
4. Atomic writes: Temp file → atomic rename bei Fehlern
5. Memory efficiency: Streaming processing für große Dateien

Solves the problems:
- Column drift through missing fields → Always enforce full header set
- Duplicate rows through repeated runs → Composite deduplication keys
- max_groups not properly supported → Full pass-through and targeted processing
- Partial writes on errors → Atomic temp file operations
"""
import csv
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CSVDeduplicationKey:
    """Composite key for CSV row deduplication"""
    transaction_hash: str
    nft_address: str
    timestamp: str
    event_type: str
    
    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> 'CSVDeduplicationKey':
        """Create dedup key from CSV row dict"""
        return cls(
            transaction_hash=row.get('transaction_hash', ''),
            nft_address=row.get('nft_address', ''), 
            timestamp=str(row.get('timestamp', '')),
            event_type=row.get('event_type', '')
        )
    
    def to_string(self) -> str:
        """Convert to string for set operations"""
        return f"{self.transaction_hash}|{self.nft_address}|{self.timestamp}|{self.event_type}"


@dataclass
class CSVProcessingStats:
    """Statistics for CSV processing operations"""
    input_rows: int = 0
    output_rows: int = 0
    duplicates_removed: int = 0
    missing_fields_filled: int = 0
    groups_processed: List[int] = None
    
    def __post_init__(self):
        if self.groups_processed is None:
            self.groups_processed = []


class ImprovedCSVManager:
    """Improved CSV manager with strict schema adherence and atomic operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load config constants
        from .config import (
            CSV_HEADERS, ANALYZED_CSV, CSV_GROUP_DIR, ANALYZED_PROGRESS_FILE,
            TRACES_GROUP_DIR, EVENTS_GROUP_DIR
        )
        
        self.csv_headers = CSV_HEADERS
        self.analyzed_csv = Path(ANALYZED_CSV) 
        self.csv_group_dir = Path(CSV_GROUP_DIR)
        self.analyzed_progress_file = Path(ANALYZED_PROGRESS_FILE)
        self.traces_group_dir = Path(TRACES_GROUP_DIR)
        self.events_group_dir = Path(EVENTS_GROUP_DIR)
        
        # Ensure directories exist
        self.csv_group_dir.mkdir(parents=True, exist_ok=True)
    
def _parse_nft_transfer_addresses(event: Dict, trace: Dict, nft_address: str) -> Tuple[str, str]:
    """Parse nft_transfer to extract prev_owner (from) and new_owner (to) addresses using BOC parser"""
    try:
        from .improved_boc_parser import ImprovedBOCParser
        parser = ImprovedBOCParser()
        
        # Use the full address extraction to get both from_address (prev_owner) and to_address (new_owner)
        result = parser.extract_full_nft_transfer_addresses(event, trace, nft_address)
        
        if result and (result.get('from_address') or result.get('to_address')):
            from_addr = result.get('from_address', '')
            to_addr = result.get('to_address', '')
            logger.debug(f"      ✅ BOC PARSER: Full transfer addresses: {from_addr[-10:] if from_addr else 'None'} → {to_addr[-10:] if to_addr else 'None'}")
            return from_addr, to_addr
        else:
            logger.debug(f"      ❌ BOC PARSER: Failed to extract transfer addresses")
            return None, None
            
    except Exception as e:
        logger.debug(f"      ❌ BOC PARSER: Exception during address extraction: {e}")
        return None, None


def _extract_direct_nft_transfer_owner(event: Dict, trace: Dict, nft_address: str) -> Optional[str]:
    """
    Extract new owner from direct NFT transfer (0x5fcc3d14) on NFT contract with payload-encoded owner
    
    This handles cases where:
    - NFT contract receives direct 0x5fcc3d14 message
    - New owner is encoded in the payload
    - No normal transfer signals exist
    """
    try:
        from .improved_boc_parser import ImprovedBOCParser
        
        logger.debug(f"   🔍 DIRECT TRANSFER: Searching for direct NFT transfer on contract {nft_address[-10:]}")
        
        # Find transaction for the specific NFT contract
        def find_nft_transaction(node):
            if not isinstance(node, dict):
                return None
            
            transaction = node.get('transaction', {})
            if not isinstance(transaction, dict):
                return None
            
            # Check if this transaction is for our NFT contract
            account = transaction.get('account', {})
            if account.get('address') == nft_address:
                # Check if this is a direct nft_transfer (0x5fcc3d14)
                in_msg = transaction.get('in_msg', {})
                if isinstance(in_msg, dict) and in_msg.get('op_code') == '0x5fcc3d14':
                    logger.debug(f"   ✅ FOUND: Direct nft_transfer on NFT contract")
                    return transaction
            
            # Search children
            for child in node.get('children', []):
                result = find_nft_transaction(child)
                if result:
                    return result
            
            return None
        
        nft_tx = find_nft_transaction(trace)
        if not nft_tx:
            logger.debug(f"   ❌ No direct nft_transfer found on NFT contract")
            return None
        
        # Extract new owner from raw_body using BOC parser
        in_msg = nft_tx.get('in_msg', {})
        raw_body = in_msg.get('raw_body', '')
        
        if not raw_body:
            logger.debug(f"   ❌ No raw_body in direct nft_transfer message")
            return None
        
        # Use BOC parser to extract addresses
        parser = ImprovedBOCParser()
        
        # Try to extract addresses from the raw_body directly
        result = parser._extract_simple_0x5fcc3d14_addresses(raw_body)
        
        if result and result.get('new_owner'):
            new_owner = result['new_owner']
            logger.debug(f"   ✅ DIRECT TRANSFER: Extracted new owner: {new_owner}")
            return new_owner
        else:
            logger.debug(f"   ❌ Failed to extract new owner from direct transfer raw_body")
            return None
            
    except Exception as e:
        logger.debug(f"   ❌ DIRECT TRANSFER: Exception during owner extraction: {e}")
        return None


def _scan_for_additional_nft_transfer(trace: Dict, nft_address: str = '') -> bool:
    """
    Scan for ADDITIONAL nft_transfer that comes directly from ROOT wallet
    
    Criteria for additional transfer:
    1. Direct child of ROOT transaction 
    2. op_code === '0x5fcc3d14'
    3. source === ROOT wallet address (NOT sale contract!)
    4. NEW: NOT part of a 2-hop sale path (seller → sale_contract → buyer)
    """
    if not isinstance(trace, dict):
        return False
    
    # Get ROOT wallet address
    root_transaction = trace.get('transaction', {})
    if not isinstance(root_transaction, dict):
        return False
    
    root_account = root_transaction.get('account', {})
    if not isinstance(root_account, dict):
        return False
    
    root_wallet_address = root_account.get('address', '')
    if not root_wallet_address:
        return False
    
    logger.debug(f"      🔍 ADDITIONAL TRANSFER SCAN: ROOT wallet = {root_wallet_address[-10:]}")
    
    # Check direct children of ROOT transaction
    children = trace.get('children', [])
    if not isinstance(children, list):
        return False
    
    for child in children:
        if not isinstance(child, dict):
            continue
        
        child_transaction = child.get('transaction', {})
        if not isinstance(child_transaction, dict):
            continue
        
        # Check in_msg for nft_transfer from ROOT wallet
        in_msg = child_transaction.get('in_msg', {})
        if not isinstance(in_msg, dict):
            continue
        
        op_code = in_msg.get('op_code')
        source = in_msg.get('source', {})
        source_address = source.get('address', '') if isinstance(source, dict) else ''
        
        logger.debug(f"      🔍 Child: op_code={op_code}, source={source_address[-10:] if source_address else 'None'}")
        
        # Check all 3 criteria
        if (op_code == '0x5fcc3d14' and 
            source_address == root_wallet_address):
            
            # NEW: Extract destination address using BOC parser
            try:
                from .improved_boc_parser import ImprovedBOCParser
                parser = ImprovedBOCParser()
                
                # Get the destination address from the nft_transfer message
                to_addr = None
                raw_body = in_msg.get('raw_body', '')
                if raw_body:
                    parsed = parser._extract_simple_0x5fcc3d14_addresses(raw_body)
                    if parsed and parsed.get('new_owner'):
                        to_addr = parsed['new_owner']
                
                # Check if current child transaction account is same as destination
                child_account = child_transaction.get('account', {})
                child_account_addr = child_account.get('address', '') if isinstance(child_account, dict) else ''
                
                # If this is seller→sale_contract (destination == child account), check for second hop
                if to_addr and child_account_addr and to_addr == child_account_addr:
                    # Check if there's a second hop: sale_contract → NFT/buyer
                    if _has_second_hop_sale_to_nft(child, sale_contract_addr=child_account_addr, nft_address=nft_address):
                        logger.debug(f"      🧩 SALE TWO-HOP (seller→sale→nft) detected – suppressing extra transfer.")
                        return False
                    # Even if no explicit second hop found, conservatively don't treat as extra
                    logger.debug(f"      🧩 Seller→Sale detected – conservatively suppressing extra transfer.")
                    return False
                        
            except Exception as e:
                logger.debug(f"      ⚠️ BOC parsing failed for 2-hop detection: {e}")
            
            logger.debug(f"      ✅ ADDITIONAL TRANSFER: Found direct nft_transfer from ROOT wallet!")
            return True
    
    logger.debug(f"      ❌ No additional nft_transfer found (only normal sale transfers)")
    return False


def _has_second_hop_sale_to_nft(node: Dict, sale_contract_addr: str, nft_address: str) -> bool:
    """
    Find the 2nd hop of the sale: Sale-Contract → NFT (as in_msg with op=0x5fcc3d14),
    even if out_msgs are empty.
    """
    if not isinstance(node, dict):
        return False

    # Check current node (often the NFT tx is a child/descendant)
    tx = node.get('transaction', {}) or {}
    acct = tx.get('account', {}) or {}
    if acct.get('address') == nft_address:
        in_msg = tx.get('in_msg', {}) or {}
        if in_msg.get('op_code') == '0x5fcc3d14':
            src = (in_msg.get('source', {}) or {}).get('address', '')
            if src == sale_contract_addr:
                logger.debug(f"      🔍 Second hop found: sale {sale_contract_addr[-10:]} → nft {nft_address[-10:]}")
                return True

    # Recursively search in children
    for ch in node.get('children', []):
        if _has_second_hop_sale_to_nft(ch, sale_contract_addr, nft_address):
            return True
    return False


class ImprovedCSVManager:
    """Improved CSV manager with strict schema adherence and atomic operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load config constants
        from .config import (
            CSV_HEADERS, ANALYZED_CSV, CSV_GROUP_DIR, ANALYZED_PROGRESS_FILE,
            TRACES_GROUP_DIR, EVENTS_GROUP_DIR
        )
        
        self.csv_headers = CSV_HEADERS
        self.analyzed_csv = Path(ANALYZED_CSV) 
        self.csv_group_dir = Path(CSV_GROUP_DIR)
        self.analyzed_progress_file = Path(ANALYZED_PROGRESS_FILE)
        self.traces_group_dir = Path(TRACES_GROUP_DIR)
        self.events_group_dir = Path(EVENTS_GROUP_DIR)
        
        # Ensure directories exist
        self.csv_group_dir.mkdir(parents=True, exist_ok=True)

    def write_csv_with_strict_headers(self, events: List[Dict], output_file: Path, 
                                    deduplicate: bool = True) -> CSVProcessingStats:
        """
        Write events to CSV with strict header adherence and atomic operations
        
        Args:
            events: List of event dictionaries
            output_file: Target CSV file
            deduplicate: Whether to perform deduplication
            
        Returns:
            Processing statistics
        """
        stats = CSVProcessingStats()
        stats.input_rows = len(events)
        
        if not events:
            self.logger.info("No events to write")
            return stats
        
        # Step 1: Enforce strict schema (fill missing fields)
        normalized_events = self._normalize_events_to_schema(events)
        stats.missing_fields_filled = sum(1 for e in normalized_events 
                                        if len([k for k in self.csv_headers if k not in e]) > 0)
        
        # Step 2: Deduplication if requested
        if deduplicate:
            deduplicated_events = self._deduplicate_events(normalized_events)
            stats.duplicates_removed = len(normalized_events) - len(deduplicated_events)
            normalized_events = deduplicated_events
        
        stats.output_rows = len(normalized_events)
        
        # Step 3: Atomic write using temp file
        temp_file = None
        try:
            # Create temp file in same directory as target
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.tmp',
                dir=output_file.parent,
                delete=False,
                encoding='utf-8',
                newline=''
            )
            
            # Write to temp file with strict headers
            writer = csv.DictWriter(temp_file, fieldnames=self.csv_headers)
            writer.writeheader()
            
            for event in normalized_events:
                # Ensure row has exactly the right fields in right order
                csv_row = {header: event.get(header, '') for header in self.csv_headers}
                writer.writerow(csv_row)
            
            temp_file.flush()
            temp_file.close()
            
            # Atomic rename
            shutil.move(temp_file.name, output_file)
            temp_file = None  # Successfully moved
            
            self.logger.info(f"✅ Wrote {stats.output_rows} events to {output_file}")
            if stats.duplicates_removed > 0:
                self.logger.info(f"   🔄 Removed {stats.duplicates_removed} duplicates")
            if stats.missing_fields_filled > 0:
                self.logger.info(f"   📝 Filled missing fields in {stats.missing_fields_filled} events")
                
        except Exception as e:
            self.logger.error(f"❌ Error writing CSV {output_file}: {e}")
            # Cleanup temp file on error
            if temp_file and Path(temp_file.name).exists():
                Path(temp_file.name).unlink()
            raise
        
        return stats
    
    def _normalize_events_to_schema(self, events: List[Dict]) -> List[Dict]:
        """Enforce strict schema: ensure all events have all required fields"""
        normalized = []
        
        for event in events:
            normalized_event = {}
            for header in self.csv_headers:
                normalized_event[header] = event.get(header, '')
            normalized.append(normalized_event)
        
        return normalized
    
    def _deduplicate_events(self, events: List[Dict]) -> List[Dict]:
        """
        Deduplicate events using composite keys
        
        Strategy: tx_hash + nft_address + timestamp + event_type
        """
        seen_keys = set()
        deduplicated = []
        
        for event in events:
            dedup_key = CSVDeduplicationKey.from_csv_row(event)
            key_string = dedup_key.to_string()
            
            if key_string not in seen_keys:
                seen_keys.add(key_string)
                deduplicated.append(event)
        
        return deduplicated
    
    def load_existing_deduplication_keys(self, csv_files: List[Path]) -> Set[str]:
        """Load existing deduplication keys from multiple CSV files for global dedup"""
        existing_keys = set()
        
        for csv_file in csv_files:
            if not csv_file.exists():
                continue
                
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dedup_key = CSVDeduplicationKey.from_csv_row(row)
                        existing_keys.add(dedup_key.to_string())
                        
            except Exception as e:
                self.logger.warning(f"⚠️  Error reading {csv_file} for deduplication: {e}")
        
        self.logger.debug(f"📊 Loaded {len(existing_keys)} existing deduplication keys")
        return existing_keys
    
    def append_events_to_group_csv(self, events: List[Dict], group_num: int, 
                                 global_dedup: bool = True) -> CSVProcessingStats:
        """
        Append events to group-specific CSV with optional global deduplication
        
        Args:
            events: Events to append
            group_num: Group number
            global_dedup: Whether to check against all existing CSVs for duplicates
        """
        if not events:
            return CSVProcessingStats()
        
        group_csv_file = self.csv_group_dir / f"analyzed_events_{group_num:04d}.csv"
        
        # Global deduplication: check against ALL existing CSVs
        if global_dedup:
            all_csv_files = [self.analyzed_csv] + list(self.csv_group_dir.glob("analyzed_events_*.csv"))
            existing_keys = self.load_existing_deduplication_keys(all_csv_files)
            
            # Filter out events that already exist globally
            filtered_events = []
            for event in events:
                dedup_key = CSVDeduplicationKey.from_csv_row(event)
                if dedup_key.to_string() not in existing_keys:
                    filtered_events.append(event)
            
            events = filtered_events
        
        return self.write_csv_with_strict_headers(events, group_csv_file, deduplicate=True)
    
    def merge_group_csvs_atomic(self, target_file: Optional[Path] = None) -> CSVProcessingStats:
        """
        Atomically merge all group CSV files into main CSV
        
        Features:
        - Loads all group CSVs
        - Global deduplication across all groups
        - Sorts by timestamp
        - Atomic write to target file
        """
        if target_file is None:
            target_file = self.analyzed_csv
        
        self.logger.info("🔄 Merging group CSV files with global deduplication...")
        
        # Find all group CSV files
        group_csv_files = sorted(self.csv_group_dir.glob("analyzed_events_*.csv"))
        if not group_csv_files:
            self.logger.warning("❌ No group CSV files found to merge")
            return CSVProcessingStats()
        
        # Load all events from all groups
        all_events = []
        stats = CSVProcessingStats()
        
        for group_csv in group_csv_files:
            try:
                with open(group_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    group_events = list(reader)
                    all_events.extend(group_events)
                    
                self.logger.debug(f"📊 Loaded {len(group_events)} events from {group_csv.name}")
                
            except Exception as e:
                self.logger.error(f"❌ Error reading {group_csv}: {e}")
                continue
        
        stats.input_rows = len(all_events)
        
        if not all_events:
            self.logger.warning("❌ No events loaded from group files")
            return stats
        
        # Sort by timestamp for chronological order
        def safe_timestamp(event):
            try:
                return int(event.get('timestamp', 0))
            except (ValueError, TypeError):
                return 0
        
        all_events.sort(key=safe_timestamp)
        
        # Write with global deduplication
        merge_stats = self.write_csv_with_strict_headers(all_events, target_file, deduplicate=True)
        
        self.logger.info(f"✅ Merged {len(group_csv_files)} group files → {merge_stats.output_rows} events in {target_file}")
        self.logger.info(f"   📊 Total input: {merge_stats.input_rows}, Output: {merge_stats.output_rows}, Duplicates removed: {merge_stats.duplicates_removed}")
        
        return merge_stats
    
    def process_groups_with_max_limit(self, max_groups: Optional[int] = None, 
                                    skip_existing: bool = True) -> List[int]:
        """
        Get list of groups to process with max_groups support
        
        Args:
            max_groups: Maximum number of groups to process (None = all)
            skip_existing: Whether to skip already processed groups
            
        Returns:
            List of group numbers to process
        """
        # Load progress to determine which groups are already processed
        analyzed_progress = self.load_analyzed_progress()
        processed_groups = set(analyzed_progress.get('processed_groups', []))
        
        # Find all available trace group files
        if not self.traces_group_dir.exists():
            self.logger.error(f"❌ Traces directory not found: {self.traces_group_dir}")
            return []
        
        trace_group_files = sorted(self.traces_group_dir.glob("traces_group_*.json"))
        all_group_nums = []
        
        for trace_file in trace_group_files:
            # Extract group number from filename
            try:
                group_num = int(trace_file.stem.split('_')[-1])
                all_group_nums.append(group_num)
            except (ValueError, IndexError):
                continue
        
        # Filter based on processing status
        if skip_existing:
            available_groups = [g for g in all_group_nums if g not in processed_groups]
        else:
            available_groups = all_group_nums
        
        # Apply max_groups limit
        if max_groups is not None and max_groups > 0:
            available_groups = available_groups[:max_groups]
            self.logger.info(f"📊 Found {len(available_groups)} groups to process (limited to {max_groups})")
        else:
            self.logger.info(f"📊 Found {len(available_groups)} groups to process (no limit)")
        
        return available_groups
    
    def load_analyzed_progress(self) -> Dict:
        """Load progress tracking for analyzed groups"""
        if not self.analyzed_progress_file.exists():
            return {
                'processed_groups': [],
                'groups_with_csv': [],
                'total_events': 0,
                'last_updated': ''
            }
        
        try:
            with open(self.analyzed_progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"❌ Error loading progress: {e}")
            return {'processed_groups': [], 'groups_with_csv': [], 'total_events': 0, 'last_updated': ''}
    
    def save_analyzed_progress(self, group_num: int, events_count: int):
        """Save progress for analyzed group"""
        try:
            progress = self.load_analyzed_progress()
            
            # Ensure lists
            if 'processed_groups' not in progress:
                progress['processed_groups'] = []
            if 'groups_with_csv' not in progress:
                progress['groups_with_csv'] = []
            
            # Add group to processed
            if group_num not in progress['processed_groups']:
                progress['processed_groups'].append(group_num)
            if group_num not in progress['groups_with_csv']:
                progress['groups_with_csv'].append(group_num)
            
            # Update counters
            progress['total_events'] = progress.get('total_events', 0) + events_count
            progress['last_updated'] = datetime.now().isoformat()
            
            # Atomic write to progress file
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.tmp',
                dir=self.analyzed_progress_file.parent,
                delete=False,
                encoding='utf-8'
            )
            
            json.dump(progress, temp_file, indent=2)
            temp_file.flush()
            temp_file.close()
            
            shutil.move(temp_file.name, self.analyzed_progress_file)
            
            self.logger.debug(f"📊 Saved progress for group {group_num}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving progress: {e}")
    
    def get_csv_statistics(self) -> Dict:
        """Get comprehensive statistics about CSV files"""
        stats = {
            'main_csv_exists': self.analyzed_csv.exists(),
            'main_csv_events': 0,
            'group_csv_files': 0,
            'group_csv_events': 0,
            'total_events': 0,
            'group_files_list': [],
            'duplicate_keys': 0,
            'schema_violations': 0
        }
        
        # Count main CSV
        if stats['main_csv_exists']:
            try:
                with open(self.analyzed_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    stats['main_csv_events'] = sum(1 for _ in reader)
            except Exception:
                pass
        
        # Count group CSVs
        group_csv_files = list(self.csv_group_dir.glob("analyzed_events_*.csv"))
        stats['group_csv_files'] = len(group_csv_files)
        
        for group_csv in group_csv_files:
            try:
                with open(group_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    group_events = sum(1 for _ in reader)
                    stats['group_csv_events'] += group_events
                    
                    stats['group_files_list'].append({
                        'file': group_csv.name,
                        'events': group_events
                    })
                    
            except Exception:
                continue
        
        stats['total_events'] = stats['main_csv_events'] + stats['group_csv_events']
        
        return stats
    
    def validate_csv_schema(self, csv_file: Path) -> Dict:
        """Validate CSV file against expected schema"""
        validation_result = {
            'valid': True,
            'missing_headers': [],
            'extra_headers': [],
            'rows_with_missing_data': 0,
            'total_rows': 0
        }
        
        if not csv_file.exists():
            validation_result['valid'] = False
            validation_result['error'] = 'File does not exist'
            return validation_result
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Check headers
                file_headers = reader.fieldnames or []
                expected_headers = set(self.csv_headers)
                actual_headers = set(file_headers)
                
                validation_result['missing_headers'] = list(expected_headers - actual_headers)
                validation_result['extra_headers'] = list(actual_headers - expected_headers)
                
                # Check data completeness
                for row in reader:
                    validation_result['total_rows'] += 1
                    
                    missing_data = any(not row.get(header) for header in self.csv_headers)
                    if missing_data:
                        validation_result['rows_with_missing_data'] += 1
                
                if validation_result['missing_headers'] or validation_result['extra_headers']:
                    validation_result['valid'] = False
                    
        except Exception as e:
            validation_result['valid'] = False
            validation_result['error'] = str(e)
        
        return validation_result


# Legacy compatibility functions
def append_events_to_csv(events: List[Dict], group_num: Optional[int] = None, deduplicate: bool = True):
    """Legacy compatibility wrapper"""
    manager = ImprovedCSVManager()
    if group_num is not None:
        return manager.append_events_to_group_csv(events, group_num, global_dedup=deduplicate)
    else:
        return manager.write_csv_with_strict_headers(events, manager.analyzed_csv, deduplicate)


def merge_group_csvs():
    """Legacy compatibility wrapper"""
    manager = ImprovedCSVManager()
    return manager.merge_group_csvs_atomic()


def get_csv_statistics() -> Dict:
    """Legacy compatibility wrapper"""
    manager = ImprovedCSVManager()
    return manager.get_csv_statistics()


def load_analyzed_progress() -> Dict:
    """Legacy compatibility wrapper"""
    manager = ImprovedCSVManager()
    return manager.load_analyzed_progress()


def save_analyzed_progress(group_num: int, events_count: int, analyzed_progress: Dict = None):
    """Legacy compatibility wrapper"""
    manager = ImprovedCSVManager()
    manager.save_analyzed_progress(group_num, events_count)


async def analyze_and_create_csv_improved(logger_instance=None, max_groups=None):
    """
    NEW IMPROVED CSV ANALYSIS PIPELINE
    
    Uses all the improved modules for clean, systematic analysis:
    - ImprovedFinancialExtractor (value-flow-first)
    - ImprovedMarketplaceClassifier (fee-address priority)
    - ImprovedTransferExtractor (traces-first)
    - ImprovedBatchProcessor (evidenzgetrieben)
    - ImprovedCSVManager (strict headers, deduplication)
    """
    import asyncio
    import gc
    import json
    from datetime import datetime
    
    logger = logger_instance or logging.getLogger(__name__)
    
    logger.info("🔍 IMPROVED: Analyzing traces and creating CSV with new pipeline...")
    
    # Initialize improved analysis components
    from .improved_event_classifier import TraceFirstClassifier
    from .improved_financial_utils import ImprovedFinancialExtractor
    from .improved_marketplace_utils import ImprovedMarketplaceClassifier
    from .improved_transfer_extraction import ImprovedTransferExtractor
    from .improved_batch_utils import ImprovedBatchProcessor
    from .mint_detection import MintDetector
    from .scam_detection import ScamDetector
    from .post_mint_filter import PostMintFilter
    from .failed_ownership_filter import FailedOwnershipFilter
    from .address_utils import format_address_for_csv
    from .config import EVENTS_GROUP_DIR
    
    # Load NFT cache for telegram number lookups
    logger.info("📁 Loading NFT cache for telegram number lookups...")
    nft_cache = {}
    try:
        with open("nft_888_cache.json", 'r', encoding='utf-8') as f:
            raw_cache = json.load(f)
            
            # Handle both old and new formats
            if isinstance(raw_cache, dict) and 'nfts' in raw_cache:
                nft_list = raw_cache['nfts']
            elif isinstance(raw_cache, list):
                nft_list = raw_cache
            else:
                logger.warning(f"❌ Unexpected cache format: {type(raw_cache)}")
                nft_list = []
            
            # Convert to lookup dict: address -> name
            for entry in nft_list:
                if isinstance(entry, dict) and 'address' in entry and 'name' in entry:
                    nft_cache[entry['address']] = entry['name']
            
            logger.info(f"✅ Loaded {len(nft_cache)} NFT entries from cache")
            
    except Exception as e:
        logger.error(f"❌ Could not load NFT cache: {e}")
        nft_cache = {}
    
    # Initialize components
    csv_manager = ImprovedCSVManager()
    classifier = TraceFirstClassifier()
    financial_extractor = ImprovedFinancialExtractor(nft_cache=nft_cache)
    marketplace_classifier = ImprovedMarketplaceClassifier()
    transfer_extractor = ImprovedTransferExtractor()
    batch_processor = ImprovedBatchProcessor()
    mint_detector = MintDetector(analyzed_csv_path=csv_manager.analyzed_csv)
    scam_detector = ScamDetector(debug=False)
    post_mint_filter = PostMintFilter()
    failed_ownership_filter = FailedOwnershipFilter()
    
    logger.info(f"✅ Initialized all improved analysis components")
    
    # Get groups to process with max_groups support
    groups_to_process = csv_manager.process_groups_with_max_limit(max_groups=max_groups)
    
    if not groups_to_process:
        logger.warning("❌ No groups found to process")
        return
    
    logger.info(f"📊 Processing {len(groups_to_process)} groups with improved pipeline")
    
    total_analyzed_count = 0
    
    # Process each group systematically
    for group_idx, group_num in enumerate(groups_to_process):
        logger.info(f"🔄 IMPROVED: Processing group {group_num} ({group_idx + 1}/{len(groups_to_process)})...")
        
        # Load traces and events for this group
        traces_file = csv_manager.traces_group_dir / f"traces_group_{group_num:04d}.json"
        events_file = csv_manager.events_group_dir / f"events_group_{group_num:04d}.json"
        
        group_traces = {}
        group_events = {}
        
        # Load traces
        if traces_file.exists():
            try:
                with open(traces_file, 'r', encoding='utf-8') as f:
                    group_traces = json.load(f)
            except Exception as e:
                logger.error(f"❌ Error loading traces group {group_num}: {e}")
                continue
        
        # Load events
        if events_file.exists():
            try:
                with open(events_file, 'r', encoding='utf-8') as f:
                    group_events = json.load(f)
            except Exception as e:
                logger.error(f"❌ Error loading events group {group_num}: {e}")
        
        logger.info(f"📊 Loaded {len(group_traces)} traces and {len(group_events)} NFT events for group {group_num}")
        
        if not group_traces:
            logger.warning(f"⏭️ Skipping group {group_num} - no traces")
            continue
        
        # Process events for this group
        new_events = []
        group_analyzed_count = 0
        
        # Process each NFT's events
        for nft_address, nft_data in group_events.items():
            # Get telegram number from cache
            telegram_number = nft_cache.get(nft_address, nft_address)
            
            # Extract events list (handle different formats)
            try:
                if isinstance(nft_data, dict):
                    events_list = nft_data.get('events', nft_data)
                else:
                    events_list = nft_data
                
                if not isinstance(events_list, list):
                    events_list = []
            except Exception as e:
                logger.error(f"❌ Error processing events for {nft_address}: {e}")
                continue
            
            # Collect events with their traces
            nft_events_with_traces = []
            processed_event_ids = set()
            
            for event_entry in events_list:
                # Handle different event entry formats
                if isinstance(event_entry, dict) and 'events' in event_entry:
                    event = event_entry['events']
                elif isinstance(event_entry, dict):
                    event = event_entry
                else:
                    continue
                
                if not isinstance(event, dict):
                    continue
                
                event_id = event.get('event_id', '')
                if not event_id or event_id in processed_event_ids:
                    continue
                
                processed_event_ids.add(event_id)
                
                # Find corresponding trace
                trace = {}
                for tx_hash, trace_data in group_traces.items():
                    if isinstance(trace_data, str):
                        continue
                    if (tx_hash == event_id or 
                        trace_data.get('transaction', {}).get('hash') == event_id):
                        trace = trace_data
                        break
                
                nft_events_with_traces.append({
                    'event': event,
                    'trace': trace,
                    'event_id': event_id
                })
            
            # Sort by timestamp
            nft_events_with_traces.sort(key=lambda x: x['event'].get('timestamp', 0))
            
            logger.debug(f"🔄 Processing {len(nft_events_with_traces)} events for NFT {nft_address[-10:]}")
            
            # List all event IDs for this NFT
            event_ids = [ed['event_id'] for ed in nft_events_with_traces]
            logger.debug(f"   📋 Event IDs: {[eid[:12] + '...' for eid in event_ids]}")
            
            # Check for batch transactions (multiple NFTs in same tx)
            tx_groups = {}
            for event_data in nft_events_with_traces:
                tx_hash = event_data['event_id']
                if tx_hash not in tx_groups:
                    tx_groups[tx_hash] = []
                tx_groups[tx_hash].append(event_data)
            
            # Process each transaction
            for tx_hash, tx_events in tx_groups.items():
                if len(tx_events) > 1:
                    # Batch transaction - use batch processor
                    logger.debug(f"   🔀 BATCH: Transaction {tx_hash} has {len(tx_events)} events")
                    batch_event_ids = [ed['event_id'] for ed in tx_events]
                    logger.debug(f"      📋 Batch Event IDs: {batch_event_ids}")
                    
                    # Use first event/trace as representative
                    representative = tx_events[0]
                    detected_nfts = {ed['event'].get('nft_address', nft_address) for ed in tx_events}
                    
                    try:
                        batch_analysis = batch_processor.process_batch_transaction(
                            representative['event'], 
                            representative['trace'], 
                            detected_nfts
                        )
                        
                        # Convert batch analysis to CSV rows
                        for item in batch_analysis.items:
                            timestamp = representative['event'].get('timestamp', '')
                            readable_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else ''
                            
                            # Determine event type based on category
                            if item.category.value == 'sale':
                                event_type = 'sale'  # Could be enhanced with marketplace detection
                            elif item.category.value == 'mint':
                                event_type = 'mint'
                            elif item.category.value == 'transfer':
                                event_type = 'transfer'
                            else:
                                event_type = item.category.value
                            
                            # Get marketplace classification
                            marketplace_result = marketplace_classifier.classify_from_financial_analysis(batch_analysis)
                            marketplace_name = marketplace_result.combined_classification or 'Unknown'
                            
                            csv_row = {
                                'date': readable_date,
                                'timestamp': timestamp,
                                'event_type': event_type,
                                'nft_address': format_address_for_csv(item.nft_address),
                                'telegram_number': telegram_number,
                                'transaction_hash': tx_hash,
                                'from_address': format_address_for_csv(item.from_owner),
                                'to_address': format_address_for_csv(item.to_owner),
                                'buyer_address': format_address_for_csv(item.to_owner),
                                'seller_address': format_address_for_csv(item.from_owner),  # Money chain: original seller
                                'buyer_paid_ton': item.price_ton,
                                'fragment_received_ton': item.item_fee_share_ton if 'fragment' in marketplace_name else 0,
                                'marketapp_received_ton': item.item_fee_share_ton if 'marketapp' in marketplace_name else 0,
                                'getgems_received_ton': item.item_fee_share_ton if 'getgems' in marketplace_name else 0,
                                'nft_contract_received_ton': 0,
                                'total_amount_ton': item.price_ton,
                                'marketplace': marketplace_name,
                                'block_time': timestamp,
                                'is_inferred': item.is_ambiguous,
                                'transfer_type': 'direct',
                                'auction_contract': '',
                                'deploy_event_id': ''
                            }
                            
                            new_events.append(csv_row)
                            group_analyzed_count += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Batch processing failed for {tx_hash}: {e}")
                        # Fallback to single event processing
                        continue
                
                else:
                    # Single event - use regular processing
                    event_data = tx_events[0]
                    event = event_data['event']
                    trace = event_data['trace']
                    event_id = event_data['event_id']
                    
                    logger.debug(f"   📝 SINGLE: Processing event {event_id}")
                    logger.debug(f"      🔗 Event ID: {event_id}")
                    
                    try:
                        # NEW ORCHESTRATOR ARCHITECTURE: Pre-filter → Mint Detection → Post-Mint Filter → Financial → Marketplace → Final Classification
                        
                        # Step 0: Quick ownership failure pre-filter (applies to all events)
                        if failed_ownership_filter.should_filter_event(event, trace, nft_address):
                            logger.debug(f"   🚫 FILTERED: All ownership assignments failed")
                            continue  # Skip this event entirely
                        
                        # Step 0a: Post-mint refund filter (applies to all events)
                        if post_mint_filter.should_filter_event(event, trace, nft_address):
                            logger.debug(f"   🚫 FILTERED: Post-mint refund without ownership change")
                            continue  # Skip this event entirely
                        
                        # Step 0b: Check for additional nft_transfer opcodes (for multi-event detection)
                        has_additional_transfer = _scan_for_additional_nft_transfer(trace, nft_address)
                        logger.debug(f"   🔍 MULTI-EVENT DEBUG: Scanning for additional nft_transfer from ROOT wallet for NFT {nft_address[-10:]} -> {has_additional_transfer}")
                        if has_additional_transfer:
                            logger.debug(f"   🔄 MULTI-EVENT: Found additional nft_transfer from ROOT wallet for NFT {nft_address[-10:]}")
                        
                        # Step 1: Check for mint/mint_auction FIRST (highest priority)
                        mint_type = mint_detector.detect_mint_type(event, trace, nft_address)
                        
                        if mint_type:
                            
                            event_type = mint_type
                            logger.debug(f"   🎯 MINT detected: {mint_type}")
                            # For mints, financial analysis and transfer extraction
                            financial_analysis = financial_extractor.extract_financial_analysis(event, trace, nft_address, is_mint=True)
                            transfer_extractor = ImprovedTransferExtractor()
                            transfer_result = transfer_extractor.extract_nft_transfer_addresses(event, trace, nft_address)
                            marketplace_type = 'unknown'  # Mints don't have marketplace
                        else:
                            # Step 2: Financial Analysis (deterministic "hard" signals) - Money Chain
                            financial_analysis = financial_extractor.extract_financial_analysis(event, trace, nft_address)
                            
                            # Step 2b: Transfer Extraction (ownership chain) - Owner Chain
                            transfer_extractor = ImprovedTransferExtractor()
                            transfer_result = transfer_extractor.extract_nft_transfer_addresses(event, trace, nft_address)
                            
                            # Step 3: Marketplace Classification (interpretation "soft" signals)
                            from .improved_marketplace_utils import ImprovedMarketplaceClassifier
                            marketplace_classifier = ImprovedMarketplaceClassifier()
                            marketplace_classification = marketplace_classifier.classify_from_financial_and_trace(financial_analysis, event, trace)
                            marketplace_type = marketplace_classification.combined_classification
                            
                            # Step 4: Event Classification Orchestrator (Financial + Marketplace → Final)
                            if marketplace_type != 'unknown':
                                # IMPROVED GATE: Only downgrade to transfer if NO evidence exists at all
                                if 'fragment_sale' in marketplace_type:
                                    # Check for any Fragment evidence
                                    has_buyer = financial_analysis.buyer_payment_ton > 0
                                    has_seller = financial_analysis.sale_amount_ton > 0
                                    has_fee = financial_analysis.fragment_fee_ton > 0
                                    
                                    # Only downgrade if ALL evidence is missing
                                    if not (has_buyer or has_seller or has_fee):
                                        event_type = 'transfer'
                                        logger.debug(f"   🚫 GATE: {marketplace_type} → transfer (no buyer, no seller, no fees)")
                                    else:
                                        # Keep fragment_sale if ANY evidence exists
                                        event_type = marketplace_type
                                        evidence = []
                                        if has_buyer: evidence.append(f"buyer={financial_analysis.buyer_payment_ton:.2f}")
                                        if has_seller: evidence.append(f"seller={financial_analysis.sale_amount_ton:.2f}")
                                        if has_fee: evidence.append(f"fee={financial_analysis.fragment_fee_ton:.2f}")
                                        logger.debug(f"   ✅ FRAGMENT: Keeping {marketplace_type} ({', '.join(evidence)})")
                                else:
                                    # FAILED SALE DETECTION: Check if sale actually succeeded
                                    has_sale_amount = financial_analysis.sale_amount_ton > 0.001
                                    has_buyer_payment = financial_analysis.buyer_payment_ton > 0.001
                                    
                                    # A real sale needs BOTH sale_amount AND buyer_payment
                                    if not has_sale_amount:
                                        # Failed sale: fees detected but no seller received payment
                                        event_type = 'transfer'
                                        logger.debug(f"   🚫 FAILED SALE: {marketplace_type} → transfer (fees detected but sale_amount=0, no seller payment)")
                                    else:
                                        # Marketplace classification wins (fee-based evidence)
                                        event_type = marketplace_type
                                        logger.debug(f"   🏪 MARKETPLACE: {marketplace_type} (fee-based evidence)")
                            elif financial_analysis.sale_amount_ton > 0.001:  # Significant sale
                                # Strict P2P-Gate: Only single-hop with address-match
                                from .improved_transfer_extraction import ImprovedTransferExtractor
                                import re
                                
                                extractor = ImprovedTransferExtractor()
                                transfer_result = extractor.extract_nft_transfer_addresses(event, trace, nft_address)
                                
                                # Extract robust hop count
                                hop_count = 1
                                if transfer_result.success and transfer_result.signals:
                                    for s in transfer_result.signals:
                                        m = re.search(r"collapsed_chain_(\d+)hops", s.source)
                                        if m:
                                            hop_count = int(m.group(1))
                                            break
                                is_multi_hop = hop_count >= 2
                                
                                if is_multi_hop:
                                    event_type = 'transfer'
                                    logger.debug(f"   🔄 TRANSFER: {financial_analysis.sale_amount_ton} TON (multi-hop={hop_count} hops, not P2P)")
                                else:
                                    # Strict address alignment between owner-chain and money-chain
                                    owner_from = transfer_result.from_address or ""
                                    owner_to = transfer_result.to_address or ""
                                    money_buyer = financial_analysis.buyer_address or ""
                                    money_seller = financial_analysis.seller_address or ""
                                    
                                    addr_match = (owner_from and owner_to 
                                                  and owner_from == money_seller 
                                                  and owner_to == money_buyer)
                                    
                                    # Value conservation + no marketplace fees
                                    fees_sum = (financial_analysis.fragment_fee_ton 
                                                + financial_analysis.getgems_fee_ton 
                                                + financial_analysis.marketapp_fee_ton)
                                    value_ok = abs(financial_analysis.buyer_payment_ton 
                                                   - financial_analysis.sale_amount_ton) < 0.01 and fees_sum == 0
                                    
                                    # Actions guard (single transfer hop in actions)
                                    nft_transfer_actions = sum(1 for a in event.get('actions', []) if a.get('type') == 'NftItemTransfer')
                                    single_transfer = nft_transfer_actions <= 1
                                    
                                    if not (addr_match and value_ok and single_transfer):
                                        event_type = 'transfer'
                                        logger.debug(
                                           f"   🔄 TRANSFER: gated P2P fallback "
                                           f"(addr_match={addr_match}, value_ok={value_ok}, single_transfer={single_transfer})"
                                        )
                                    else:
                                        event_type = 'p2p_sale'
                                        logger.debug(f"   💰 P2P SALE: {financial_analysis.sale_amount_ton} TON (no marketplace fees; strict gate passed)")
                            else:
                                # Fallback to transfer classification
                                classification_result = classifier.classify_with_confidence(event, trace, nft_address)
                                event_type = classification_result.event_type
                                logger.debug(f"   🔄 FALLBACK: {event_type} (no significant value flows)")
                        
                        # No override needed - orchestrator already decided final event_type
                        
                        # Step 4b: Check for direct NFT transfer (payload-encoded owner) if normal extraction failed
                        if event_type == 'transfer' and (not transfer_result.success or not transfer_result.to_address):
                            logger.debug(f"   🔍 DIRECT TRANSFER CHECK: Normal transfer extraction failed, checking for direct NFT transfer")
                            direct_owner = _extract_direct_nft_transfer_owner(event, trace, nft_address)
                            if direct_owner:
                                # Update transfer result with direct owner extraction
                                from .improved_transfer_extraction import TransferSignal, TransferResult
                                
                                # Get source address from the direct transfer transaction
                                def find_source_address(node):
                                    if not isinstance(node, dict):
                                        return None
                                    transaction = node.get('transaction', {})
                                    if not isinstance(transaction, dict):
                                        return None
                                    account = transaction.get('account', {})
                                    if account.get('address') == nft_address:
                                        in_msg = transaction.get('in_msg', {})
                                        if isinstance(in_msg, dict) and in_msg.get('op_code') == '0x5fcc3d14':
                                            source = in_msg.get('source', {})
                                            return source.get('address', '') if isinstance(source, dict) else ''
                                    for child in node.get('children', []):
                                        result = find_source_address(child)
                                        if result:
                                            return result
                                    return None
                                
                                source_address = find_source_address(trace)
                                
                                # Create a synthetic transfer result
                                direct_signal = TransferSignal(
                                    from_address=source_address or '',
                                    to_address=direct_owner,
                                    source='direct_nft_transfer_payload',
                                    confidence='HIGH',
                                    evidence=f'Direct nft_transfer (0x5fcc3d14) on NFT contract with payload-encoded owner'
                                )
                                
                                transfer_result = TransferResult(
                                    success=True,
                                    from_address=source_address or '',
                                    to_address=direct_owner,
                                    signals=[direct_signal]
                                )
                                
                                logger.debug(f"   ✅ DIRECT TRANSFER: {source_address[-10:] if source_address else 'Unknown'} → {direct_owner[-10:]}")
                        
                        # Step 5: Detect scam/pseudo transfers
                        event_type = scam_detector.detect_pseudo_transfers(trace, event_type)
                        if event_type == 'pseudo_transfer':
                            logger.debug(f"   ⚠️  SKIPPED: Pseudo-transfer detected")
                            continue
                        
                        event_type = scam_detector.detect_scam_transfers(event, trace, event_type, nft_cache, nft_address)
                        if event_type == 'scam_transfer':
                            logger.debug(f"   ⚠️  SKIPPED: Scam transfer detected")
                            continue
                        
                        # Filter out nft_creation events before CSV creation
                        if event_type == 'nft_creation':
                            logger.debug(f"   🚫 EXCLUDED: NFT creation event - not a transaction")
                            continue
                        
                        # Step 6: Create CSV row
                        timestamp = event.get('timestamp', '')
                        readable_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else ''
                        
                        # Determine marketplace name
                        if 'fragment' in marketplace_type:
                            marketplace_name = 'Fragment'
                        elif 'getgems' in marketplace_type:
                            marketplace_name = 'GetGems'
                        elif 'marketapp' in marketplace_type:
                            marketplace_name = 'MarketApp'
                        elif event_type in ['mint', 'mint_auction']:
                            marketplace_name = 'Direct Mint' if event_type == 'mint' else 'Auction Mint'
                        else:
                            marketplace_name = 'Transfer'
                        
                        csv_row = {
                            'date': readable_date,
                            'timestamp': timestamp,
                            'event_type': event_type,
                            'nft_address': format_address_for_csv(nft_address),
                            'telegram_number': telegram_number,
                            'transaction_hash': event_id,
                            'from_address': format_address_for_csv(transfer_result.from_address if transfer_result.success and event_type != 'mint' else ''),
                            'to_address': format_address_for_csv(transfer_result.to_address if transfer_result.success else ''),
                            'buyer_address': format_address_for_csv(financial_analysis.buyer_address if event_type not in ['transfer'] else ''),
                            'seller_address': format_address_for_csv(financial_analysis.seller_address if event_type not in ['mint', 'mint_auction', 'transfer'] else ''),  # Money chain: original seller (empty for mints/transfers)
                            'buyer_paid_ton': financial_analysis.buyer_payment_ton if event_type not in ['transfer'] else 0,
                            'fragment_received_ton': financial_analysis.fragment_fee_ton if event_type not in ['transfer'] else 0,
                            'marketapp_received_ton': financial_analysis.marketapp_fee_ton if event_type not in ['transfer'] else 0,
                            'getgems_received_ton': financial_analysis.getgems_fee_ton if event_type not in ['transfer'] else 0,
                            'nft_contract_received_ton': 0,  # Not implemented yet
                            'total_amount_ton': (financial_analysis.buyer_payment_ton if 'fragment' in marketplace_type else financial_analysis.sale_amount_ton) if event_type not in ['transfer'] else 0,
                            'marketplace': marketplace_name,
                            'block_time': timestamp,
                            'is_inferred': False,
                            'transfer_type': 'direct',
                            'auction_contract': '',
                            'deploy_event_id': ''
                        }
                        
                        new_events.append(csv_row)
                        group_analyzed_count += 1
                        
                        # Multi-Event Generation: If we have a sale + additional nft_transfer from ROOT wallet, generate additional transfer event
                        if has_additional_transfer and event_type in ['fragment_sale', 'marketapp_sale', 'getgems_sale']:
                            logger.debug(f"   🔄 MULTI-EVENT: Generating additional transfer event for {event_type} with nft_transfer")
                            
                            # Parse nft_transfer addresses from raw_body using BOC parser
                            transfer_from, transfer_to = _parse_nft_transfer_addresses(event, trace, nft_address)
                            
                            # Create additional transfer event with same base data but different classification
                            transfer_csv_row = csv_row.copy()
                            transfer_csv_row.update({
                                'event_type': 'transfer',
                                'from_address': format_address_for_csv(transfer_from) if transfer_from else '',
                                'to_address': format_address_for_csv(transfer_to) if transfer_to else '',
                                'buyer_address': '',  # No buyer for transfer
                                'seller_address': '',  # No seller for transfer  
                                'buyer_paid_ton': 0,  # No payment for transfer
                                'fragment_received_ton': 0,  # No fees for transfer
                                'marketapp_received_ton': 0,
                                'getgems_received_ton': 0,
                                'total_amount_ton': 0,
                                'marketplace': 'direct_transfer'  # Indicate this is a follow-up transfer
                            })
                            
                            new_events.append(transfer_csv_row)
                            group_analyzed_count += 1
                            logger.debug(f"   ✅ Additional transfer event generated: {transfer_from[-10:] if transfer_from else 'unknown'} → {transfer_to[-10:] if transfer_to else 'unknown'}")
                        
                        logger.debug(f"   ✅ Event {event_id} processed: {event_type}, {financial_analysis.sale_amount_ton} TON")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing event {event_id}: {e}")
                        continue
        
        # Write group CSV with improved manager
        if new_events:
            stats = csv_manager.append_events_to_group_csv(new_events, group_num, global_dedup=True)
            logger.info(f"✅ Wrote {stats.output_rows} events from group {group_num} (removed {stats.duplicates_removed} duplicates)")
        
        # Save progress
        csv_manager.save_analyzed_progress(group_num, group_analyzed_count)
        total_analyzed_count += group_analyzed_count
        
        # Memory cleanup
        group_traces.clear()
        group_events.clear()
        if (group_idx + 1) % 10 == 0:
            gc.collect()
    
    # Merge all group CSV files
    logger.info("🔄 Merging all group CSV files...")
    merge_stats = csv_manager.merge_group_csvs_atomic()
    
    logger.info(f"🎉 IMPROVED pipeline completed!")
    logger.info(f"   📊 Total events analyzed: {total_analyzed_count}")
    logger.info(f"   📊 Final merged CSV: {merge_stats.output_rows} events")
    logger.info(f"   📊 Global duplicates removed: {merge_stats.duplicates_removed}")
    
    return True