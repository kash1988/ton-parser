"""
IMPROVED Financial Extraction System
VALUE-FLOW-FIRST approach with deterministic buyer/seller assignment

DESIGN PRINCIPLES:
1. Always aggregate from value_flow: Net per address, sum In/Out
2. Seller = largest positive net flow, Buyer = largest negative net flow  
3. Marketplace fees = Sum of positive flows to known fee/sales accounts (whitelist)
4. Consistency rules: Sum(positive) ≈ Seller+Fees+(other positive) - Flag if strongly deviates

Solves the problems:
- No more guessing amounts from comments/heuristics → Always from value_flow aggregation
- Clean fee separation → Whitelist-based fee account detection
- Deterministic buyer/seller → Largest net flows determine roles
"""
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from .address_utils import are_addresses_equal


@dataclass
class FeeCollectionConfig:
    """Configuration for unified fee collection"""
    mode: str  # 'strict', 'normal', 'relaxed'
    start_lt: int
    end_lt: int
    allowed_nodes: Optional[Set[str]] = None
    sale_contract_addr: Optional[str] = ""
    use_allowed_nodes: bool = False
    within_scope: bool = True


class UnifiedFeeCollector:
    """Unified fee collection - ersetzt 4 verschiedene _collect_fee_hits_* Methoden"""
    
    def __init__(self, logger, marketplace_addresses, normalize_fn):
        self.logger = logger
        self.marketplace_addresses = marketplace_addresses
        self.normalize_fn = normalize_fn
    
    def collect_fees(self, trace: Dict, config: FeeCollectionConfig) -> List[Dict]:
        """Eine Methode für alle Fee Collection Modi"""
        
        if config.mode == 'relaxed':
            return self._collect_relaxed(trace, config)
        elif config.mode == 'nft_specific':
            return self._collect_nft_specific(trace, config)
        else:  # 'normal' or 'strict'
            return self._collect_original(trace, config)
    
    def _collect_relaxed(self, trace: Dict, config: FeeCollectionConfig) -> List[Dict]:
        """Relaxed fee collection mode"""
        fee_hits = []
        sale_contract_normalized = self.normalize_fn(config.sale_contract_addr)
        
        def visit_relaxed_node(node):
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
                
            lt = tx.get('lt', 0)
            if not (config.start_lt <= lt <= config.end_lt):
                return
                
            account_addr = tx.get('account', {}).get('address', '')
            account_normalized = self.normalize_fn(account_addr)
            
            # Skip sale contract in relaxed mode
            if account_normalized == sale_contract_normalized:
                return
                
            if account_normalized in {self.normalize_fn(addr) for addr in self.marketplace_addresses}:
                credit_phase = tx.get('credit_phase', {})
                if 'credit' in credit_phase:
                    try:
                        credit = int(credit_phase['credit'])
                        if credit > 0:
                            fee_hits.append({
                                'address': account_addr,
                                'amount_nanoton': credit,
                                'lt': lt,
                                'flow_type': 'marketplace_fee'
                            })
                    except (ValueError, TypeError):
                        pass
        
        # UnifiedTraversal will be defined later, for now use a placeholder
        # UnifiedTraversal.simple_traverse(trace, visit_relaxed_node)
        self._traverse_simple(trace, visit_relaxed_node)
        return fee_hits
    
    def _collect_nft_specific(self, trace: Dict, config: FeeCollectionConfig) -> List[Dict]:
        """NFT-specific fee collection mode"""
        fee_hits = []
        
        def scan_node_messages(node):
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
                
            lt = tx.get('lt', 0)
            if not (config.start_lt <= lt <= config.end_lt):
                return
                
            # Additional NFT-specific filtering logic would go here
            account_addr = tx.get('account', {}).get('address', '')
            account_normalized = self.normalize_fn(account_addr)
            
            if account_normalized in {self.normalize_fn(addr) for addr in self.marketplace_addresses}:
                credit_phase = tx.get('credit_phase', {})
                if 'credit' in credit_phase:
                    try:
                        credit = int(credit_phase['credit'])
                        if credit > 0:
                            fee_hits.append({
                                'address': account_addr,
                                'amount_nanoton': credit,
                                'lt': lt,
                                'flow_type': 'marketplace_fee'
                            })
                    except (ValueError, TypeError):
                        pass
        
        self._traverse_simple(trace, scan_node_messages)
        return fee_hits
    
    def _collect_original(self, trace: Dict, config: FeeCollectionConfig) -> List[Dict]:
        """Original fee collection mode"""
        fee_hits = []
        
        def visit_node(node):
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
                
            lt = tx.get('lt', 0)
            if not (config.start_lt <= lt <= config.end_lt):
                return
                
            account_addr = tx.get('account', {}).get('address', '')
            account_normalized = self.normalize_fn(account_addr)
            
            if account_normalized in {self.normalize_fn(addr) for addr in self.marketplace_addresses}:
                credit_phase = tx.get('credit_phase', {})
                if 'credit' in credit_phase:
                    try:
                        credit = int(credit_phase['credit'])
                        if credit > 0:
                            fee_hits.append({
                                'address': account_addr,
                                'amount_nanoton': credit,
                                'lt': lt,
                                'flow_type': 'marketplace_fee'
                            })
                    except (ValueError, TypeError):
                        pass
        
        self._traverse_simple(trace, visit_node)
        return fee_hits
    
    def _traverse_simple(self, node: Dict, callback):
        """Simple traversal for fee collection"""
        callback(node)
        for child in node.get('children', []):
            self._traverse_simple(child, callback)


class UtilityHelpers:
    """Zentrale Utility-Funktionen für Financial Extraction"""
    
    @staticmethod
    def normalize_address(addr: str) -> str:
        """Eine definitive Address-Normalisierung"""
        if not addr:
            return ""
        return addr.strip().lower()
    
    @staticmethod
    def get_out_msgs(tx: Dict) -> List[Dict]:
        """Einheitliche Out-Message Extraktion"""
        out_msgs = tx.get('out_msgs', [])
        if not isinstance(out_msgs, list):
            out_msgs = [out_msgs] if out_msgs else []
        return out_msgs
    
    @staticmethod
    def classify_flow_type(address: str, nft_address: str, sale_contract_addr: str, 
                          marketplace_addresses: List[str], normalize_fn,
                          enhanced_context: Dict = None) -> str:
        """Einheitliche Flow-Type Classification"""
        addr_normalized = normalize_fn(address)
        
        # Check if it's the sale contract itself (enhanced feature)
        if sale_contract_addr and normalize_fn(sale_contract_addr) == addr_normalized:
            return 'sale_contract_credit'
        
        # Check marketplace addresses from config (highest priority)
        if addr_normalized in {normalize_fn(addr) for addr in marketplace_addresses}:
            return 'marketplace_fee'
        
        # Check if it's the NFT contract itself
        if normalize_fn(nft_address) == addr_normalized:
            return 'nft_contract'
        
        # Enhanced buyer change detection if context provided
        if enhanced_context:
            buyer_addr_norm = normalize_fn(enhanced_context.get('detected_buyer_wallet', ''))
            if buyer_addr_norm and addr_normalized == buyer_addr_norm:
                # Additional logic for buyer change vs seller payout detection
                current_credit = enhanced_context.get('current_flow_credit', 0) / 1e9
                buyer_payment = enhanced_context.get('buyer_payment', 0) / 1e9
                marketplace_fees = enhanced_context.get('marketplace_fee_sum', 0)
                
                if marketplace_fees == 0:
                    fragment_fees = enhanced_context.get('fragment_fees', 0)
                    if fragment_fees > 0:
                        marketplace_fees = fragment_fees / 1e9
                
                if buyer_payment > 0 and marketplace_fees > 0:
                    expected_seller_amount = buyer_payment - marketplace_fees
                    tolerance = max(0.05, expected_seller_amount * 0.02)
                    
                    if abs(current_credit - expected_seller_amount) <= tolerance:
                        return 'seller_credit'
                
                return 'buyer_change'
        
        # Default to seller credit for wallets
        return 'seller_credit'


class FragmentDetector:
    """
    Zentrale Fragment Sales Detection - eliminiert 6 duplizierte Methoden
    
    Ersetzt:
    - _extract_teleitem_bid_amount()
    - _extract_bid_amount_from_teleitem() 
    - _extract_buyer_from_teleitem_notification()
    - _find_fragment_auction_fillup_payment()
    - _check_auction_fillup_recipient()
    - _get_auction_fillup_amount()
    """
    
    @staticmethod
    def find_teleitem_bid_info(trace: Dict, logger=None) -> Dict:
        """
        Einheitliche TeleitemBidInfo Extraction - ersetzt 3 Methoden
        
        Returns:
            {
                "found": bool,
                "bid_amount_nanoton": int,
                "bid_amount_ton": float, 
                "buyer_address": str,
                "transaction_lt": int
            }
        """
        result = {
            "found": False,
            "bid_amount_nanoton": 0,
            "bid_amount_ton": 0.0,
            "buyer_address": "",
            "transaction_lt": 0
        }
        
        if logger:
            logger.debug(f"   🔍 FragmentDetector: Looking for TeleitemBidInfo in trace...")
        
        def scan_node(node):
            if not isinstance(node, dict):
                return None
                
            tx = node.get('transaction', {})
            if not tx:
                return None
            
            # Method 1: Check for TeleitemBidInfo in decoded_body directly
            decoded_body = tx.get('in_msg', {}).get('decoded_body', {})
            if 'TeleitemBidInfo' in decoded_body:
                bid_info = decoded_body['TeleitemBidInfo']
                if isinstance(bid_info, dict) and 'bid' in bid_info:
                    try:
                        bid_amount = int(bid_info['bid'])
                        if bid_amount > 0:
                            if logger:
                                logger.debug(f"   💰 FragmentDetector: Found TeleitemBidInfo bid: {bid_amount/1e9:.6f} TON")
                            return {
                                "found": True,
                                "bid_amount_nanoton": bid_amount,
                                "bid_amount_ton": bid_amount / 1e9,
                                "buyer_address": "",  # Not available in this method
                                "transaction_lt": tx.get('lt', 0)
                            }
                    except (ValueError, TypeError):
                        pass
            
            # Method 2: Check for nft_ownership_assigned with TeleitemBidInfo
            in_msg = tx.get('in_msg', {})
            if in_msg:
                decoded_op_name = in_msg.get('decoded_op_name', '')
                if decoded_op_name == 'nft_ownership_assigned':
                    # Check for TeleitemBidInfo in forward_payload
                    forward_payload = decoded_body.get('forward_payload', {})
                    if forward_payload:
                        value = forward_payload.get('value', {})
                        if value and value.get('sum_type', '') == 'TeleitemBidInfo':
                            # Extract buyer address from destination
                            destination = in_msg.get('destination', {})
                            buyer_address = destination.get('address', '')
                            
                            # Try to extract bid amount
                            bid_data = value.get('bid', {})
                            if bid_data and isinstance(bid_data, dict):
                                try:
                                    bid_amount = int(bid_data.get('amount', 0))
                                    if bid_amount > 0 and buyer_address:
                                        if logger:
                                            logger.debug(f"   💰 FragmentDetector: Found TeleitemBidInfo in nft_ownership_assigned: {bid_amount/1e9:.6f} TON → {buyer_address[-20:]}")
                                        return {
                                            "found": True,
                                            "bid_amount_nanoton": bid_amount,
                                            "bid_amount_ton": bid_amount / 1e9,
                                            "buyer_address": buyer_address,
                                            "transaction_lt": tx.get('lt', 0)
                                        }
                                except (ValueError, TypeError):
                                    pass
            
            return None
        
        # Use unified traversal to find TeleitemBidInfo
        found_result = UnifiedTraversal.find_first(trace, scan_node)
        if found_result:
            result.update(found_result)
            
        return result
    
    @staticmethod
    def find_auction_fillup(trace: Dict, target_address: str = "", logger=None) -> Dict:
        """
        Einheitliche auction_fill_up Detection - ersetzt 3 Methoden
        
        Args:
            target_address: Optional filter for specific recipient address
            
        Returns:
            {
                "found": bool,
                "credit_amount_nanoton": int,
                "credit_amount_ton": float,
                "recipient_address": str,
                "transaction_lt": int,
                "sale_contract_candidates": set  # For compatibility
            }
        """
        result = {
            "found": False,
            "credit_amount_nanoton": 0,
            "credit_amount_ton": 0.0,
            "recipient_address": "",
            "transaction_lt": 0,
            "sale_contract_candidates": set()
        }
        
        if logger:
            logger.debug(f"   🔍 FragmentDetector: Scanning for auction_fill_up operations...")
        
        def scan_node(node):
            if not isinstance(node, dict):
                return None
            
            tx = node.get('transaction', {})
            if not tx:
                return None
            
            # Look for auction_fill_up operation
            decoded_op_name = tx.get('in_msg', {}).get('decoded_op_name', '')
            if decoded_op_name == 'auction_fill_up':
                # Extract credit amount from this transaction
                credit_phase = tx.get('credit_phase', {})
                if 'credit' in credit_phase:
                    try:
                        credit = int(credit_phase['credit'])
                        if credit > 0:
                            account_addr = tx.get('account', {}).get('address', '')
                            
                            # If target_address specified, filter by it
                            if target_address:
                                if UtilityHelpers.normalize_address(account_addr) != UtilityHelpers.normalize_address(target_address):
                                    return None
                            
                            if logger:
                                logger.debug(f"   🎯 FragmentDetector: Found auction_fill_up: {credit/1e9:.6f} TON at {account_addr[-20:] if account_addr else 'unknown'}")
                            
                            return {
                                "found": True,
                                "credit_amount_nanoton": credit,
                                "credit_amount_ton": credit / 1e9,
                                "recipient_address": account_addr,
                                "transaction_lt": tx.get('lt', 0),
                                "sale_contract_candidates": {account_addr} if account_addr else set()
                            }
                    except (ValueError, TypeError):
                        pass
            
            return None
        
        # Use unified traversal to find auction_fill_up
        found_result = UnifiedTraversal.find_first(trace, scan_node)
        if found_result:
            result.update(found_result)
            
        return result


def _take_budget(remaining_budget: dict, want: int) -> int:
    """Klemmt den abzuführenden Betrag hart an das verbleibende Budget."""
    avail = max(0, int(remaining_budget.get("amount", 0)))
    take = min(avail, max(0, int(want)))
    remaining_budget["amount"] = avail - take
    return take
from .improved_transfer_extraction import ImprovedTransferExtractor

logger = logging.getLogger(__name__)

# ============== PHASE 1: STATE MANAGEMENT CONSOLIDATION ==============

@dataclass
class ExtractionState:
    """
    Zentrale State-Verwaltung für Financial Extraction
    Ersetzt alle 43+ self._last_* und self._current_* Variablen
    """
    # Path & Flow State
    path_flows: List[Dict] = field(default_factory=list)
    buyer_payment_nanoton: int = 0
    sale_contract_addr: str = ""
    sale_contract_address: str = ""  # Legacy alias
    
    # Indices (werden einmal gebaut, dann wiederverwendet)
    by_dest_index: Dict[str, List] = field(default_factory=dict)  # Legacy - wird ersetzt
    by_source_index: Dict[str, List] = field(default_factory=dict)  # Legacy - wird ersetzt
    trace_indices: Optional['TraceIndices'] = None  # PHASE 2: Unified indices
    
    # Analysis Context
    current_analysis_scope: Optional['AnalysisScope'] = None
    current_event_id: str = ""
    current_event: Dict = field(default_factory=dict)
    current_nft_address: str = ""
    
    # Fragment Detection & Handling
    fragment_nft_payment_detected: bool = False
    has_seller_credits: bool = False
    
    # Processing Flags
    is_mint_event: bool = False
    
    # Complex State for Advanced Processing
    seen_fee_keys: Set[str] = field(default_factory=set)
    teleitem_bid_amount: int = 0
    teleitem_buyer_payment: int = 0
    force_buyer_payment: bool = False
    
    def reset_for_new_extraction(self):
        """Reset state für neue Extraction (keeping indices)"""
        self.path_flows = []
        self.buyer_payment_nanoton = 0
        self.sale_contract_addr = ""
        self.sale_contract_address = ""
        self.current_analysis_scope = None
        self.current_event_id = ""
        self.current_event = {}
        self.current_nft_address = ""
        self.fragment_nft_payment_detected = False
        self.has_seller_credits = False
        self.is_mint_event = False
        self.seen_fee_keys = set()
        self.teleitem_bid_amount = 0
        self.teleitem_buyer_payment = 0
        self.force_buyer_payment = False
        # Indices bleiben erhalten für Performance


# ============== PHASE 2: INDEX CONSOLIDATION ==============

class TraceIndices:
    """
    Unified index building - alle Indices in einem Trace-Durchgang
    Ersetzt 3 verschiedene _build_*_index Methoden mit mehrfachen Traversals
    """
    def __init__(self, trace: Dict, normalize_fn):
        self.normalize = normalize_fn
        
        # All indices built in single traversal
        self.by_source: Dict[str, List[Dict]] = defaultdict(list)  # source_addr -> [child_nodes]
        self.by_dest: Dict[str, List[Dict]] = defaultdict(list)    # dest_addr -> [sender_nodes] 
        self.by_account: Dict[str, List[Dict]] = defaultdict(list) # account_addr -> [nodes]
        self.by_lt: Dict[int, List[Dict]] = defaultdict(list)      # lt -> [nodes]
        self.all_nodes: List[Dict] = []             # flat list of all nodes
        self.all_addresses: Set[str] = set()        # all unique addresses
        
        # Build all indices in single traversal
        self._build_all_indices(trace)
        
    def _build_all_indices(self, node: Dict):
        """Single traversal to build all indices"""
        if not isinstance(node, dict):
            return
            
        # Add to all_nodes
        self.all_nodes.append(node)
        
        tx = node.get('transaction', {})
        if not tx:
            # Recurse to children even without transaction
            for child in node.get('children', []):
                self._build_all_indices(child)
            return
            
        # Extract account address
        account_addr = self.normalize(tx.get('account', {}).get('address', ''))
        if account_addr:
            self.all_addresses.add(account_addr)
            
            # Index by account
            self.by_account[account_addr].append(node)
            
        # Index by LT
        lt = tx.get('lt', 0)
        if lt:
            self.by_lt[lt].append(node)
        
        # Index by source (in_msg.source -> this node)
        in_msg = tx.get('in_msg', {})
        if in_msg:
            source = in_msg.get('source', {})
            if source:
                source_addr = self.normalize(source.get('address', ''))
                if source_addr:
                    self.all_addresses.add(source_addr)
                    self.by_source[source_addr].append(node)
        
        # Index by destination (out_msgs.destination -> this node as sender)
        out_msgs = tx.get('out_msgs', [])
        if not isinstance(out_msgs, list):
            out_msgs = [out_msgs] if out_msgs else []
            
        for out_msg in out_msgs:
            if isinstance(out_msg, dict):
                dest = out_msg.get('destination', {})
                if dest:
                    dest_addr = self.normalize(dest.get('address', ''))
                    if dest_addr and account_addr:
                        self.all_addresses.add(dest_addr)
                        self.by_dest[dest_addr].append(node)
        
        # Recurse to children
        for child in node.get('children', []):
            self._build_all_indices(child)
    
    def get_children_by_source(self, source_addr: str) -> List[Dict]:
        """Get child nodes by source address"""
        return self.by_source[self.normalize(source_addr)]
    
    def get_senders_to_dest(self, dest_addr: str) -> List[Dict]:
        """Get sender nodes that send to destination address"""
        return self.by_dest[self.normalize(dest_addr)]
    
    def get_nodes_by_account(self, account_addr: str) -> List[Dict]:
        """Get all nodes for specific account"""
        return self.by_account[self.normalize(account_addr)]
        
    def get_nodes_by_lt_range(self, start_lt: int, end_lt: int) -> List[Dict]:
        """Get nodes in LT range"""
        nodes = []
        for lt in range(start_lt, end_lt + 1):
            nodes.extend(self.by_lt[lt])
        return nodes
    
    def stats(self) -> str:
        """Debug statistics"""
        return (f"Indices: {len(self.by_source)} sources, {len(self.by_dest)} dests, "
                f"{len(self.by_account)} accounts, {len(self.all_nodes)} nodes")


# ============== PHASE 3: TRAVERSAL UNIFICATION ==============

class UnifiedTraversal:
    """
    Unified traversal operations - ersetzt 6 verschiedene _traverse_* Methoden
    Alle Traversal-Muster in einer flexiblen API
    """
    
    @staticmethod
    def simple_traverse(start_node: Dict, callback, filter_fn=None):
        """
        Basic traversal - ersetzt _traverse_trace_tree
        """
        def visit(node):
            if not isinstance(node, dict):
                return
            
            # Apply filter if provided
            if filter_fn and not filter_fn(node):
                return
                
            # Apply callback
            callback(node)
            
            # Recurse to children
            for child in node.get('children', []):
                visit(child)
        
        visit(start_node)
    
    @staticmethod
    def find_first(start_node: Dict, condition, filter_fn=None):
        """
        Search traversal - ersetzt _traverse_trace_tree_with_result
        Returns first node where condition returns non-None
        """
        def search(node):
            if not isinstance(node, dict):
                return None
            
            # Apply filter if provided
            if filter_fn and not filter_fn(node):
                return None
                
            # Try condition on current node
            result = condition(node)
            if result:
                return result
            
            # Search children
            for child in node.get('children', []):
                result = search(child)
                if result:
                    return result
                    
            return None
        
        return search(start_node)
    
    @staticmethod
    def find_first_node(start_node: Dict, condition, filter_fn=None):
        """
        Search traversal - returns the node (not the condition result)
        Returns first node where condition returns True
        """
        def search(node):
            if not isinstance(node, dict):
                return None
            
            # Apply filter if provided
            if filter_fn and not filter_fn(node):
                return None
                
            # Try condition on current node
            if condition(node):
                return node
            
            # Search children
            for child in node.get('children', []):
                result = search(child)
                if result:
                    return result
                    
            return None
        
        return search(start_node)
    
    @staticmethod
    def collect_flows(start_node: Dict, collector, 
                     budget_tracker=None, allowed_nodes=None, 
                     lt_window=None, indices=None):
        """
        Complex flow collection - ersetzt _traverse_outgoing_flows_* Varianten
        
        Args:
            collector: Function that extracts flows from node
            budget_tracker: Dict with 'amount' key for budget tracking
            allowed_nodes: Set of allowed node IDs  
            lt_window: Tuple (start_lt, end_lt) for LT filtering
            indices: TraceIndices for optimized lookups
        """
        flows = []
        visited = set()
        
        def collect_from_node(node):
            if not isinstance(node, dict):
                return
                
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            
            tx = node.get('transaction', {})
            if not tx:
                # Still traverse children without transaction
                for child in node.get('children', []):
                    collect_from_node(child)
                return
            
            # Apply allowed nodes filter
            if allowed_nodes and node_id not in allowed_nodes:
                return
                
            # Apply LT window filter
            if lt_window:
                tx_lt = tx.get('lt', 0)
                start_lt, end_lt = lt_window
                if not (start_lt <= tx_lt <= end_lt):
                    return
            
            # Check budget before processing
            if budget_tracker and budget_tracker.get('amount', 0) <= 0:
                return
                
            # Collect flows from this node
            node_flows = collector(node, budget_tracker)
            if node_flows:
                flows.extend(node_flows)
            
            # Continue to children (use indices if available)
            if indices:
                account_addr = tx.get('account', {}).get('address', '')
                if account_addr:
                    children = indices.get_children_by_source(account_addr)
                    for child in children:
                        collect_from_node(child)
            else:
                for child in node.get('children', []):
                    collect_from_node(child)
        
        collect_from_node(start_node)
        return flows
    
    @staticmethod
    def traverse_with_context(start_node: Dict, processor, context=None):
        """
        Contextual traversal for complex processing
        processor receives (node, context) and can modify context
        """
        if context is None:
            context = {}
            
        def process_node(node):
            if not isinstance(node, dict):
                return
                
            # Process current node with context
            processor(node, context)
            
            # Recurse to children
            for child in node.get('children', []):
                process_node(child)
        
        process_node(start_node)
        return context


@dataclass 
class NetFlow:
    """Net flow for an address"""
    address: str
    net_amount_ton: float
    total_in_ton: float 
    total_out_ton: float
    is_marketplace_fee_account: bool = False
    marketplace_type: str = ""
    flow_types: List[str] = field(default_factory=list)  # Track all flow types for this address


@dataclass
class FeeHit:
    """Evidence of marketplace fee payment (amount-independent)"""
    address: str
    marketplace_type: str  # 'fragment', 'marketapp', 'getgems'
    amount_ton: float
    amount_nanoton: int
    lt: int
    aborted: bool
    evidence: str
    within_scope: bool = True  # Was this hit found within the strict NFT scope?
    linkage: str = "unknown"   # How is this connected: "source=sale_contract", "child_of_sale", etc.


@dataclass
class AnalysisScope:
    """Analysis scope for NFT-specific processing"""
    allowed_node_ids: set
    start_lt: int
    end_lt: int
    sale_contract_addr: str
    nft_address: str
    
    # Additional fields for marketplace utils
    anchors: Dict[str, str] = None  # {sale_contract, nft_item, buyer_wallet, seller_wallet}
    path_nodes: List[Dict[str, Any]] = None  # [{"address": addr, "lt": lt}, ...]
    event_id: str = ""
    scope_strategy: str = "single"  # "single" | "batch"


@dataclass
class FinancialAnalysis:
    """Complete financial analysis result"""
    seller_address: str = ""
    buyer_address: str = ""
    sale_amount_ton: float = 0.0  # What seller received (buyer payment - fees)
    buyer_payment_ton: float = 0.0  # What buyer actually paid (total outgoing)
    
    # Marketplace fees by type
    fragment_fee_ton: float = 0.0
    marketapp_fee_ton: float = 0.0
    getgems_fee_ton: float = 0.0
    
    # NEW: Raw fee evidence for marketplace classification
    fee_hits: List[FeeHit] = None
    
    # NEW: Analysis scope for scoped processing
    scope: AnalysisScope = None
    
    # Total amounts
    total_positive_flow_ton: float = 0.0
    total_negative_flow_ton: float = 0.0
    
    # Consistency checks
    is_consistent: bool = True
    consistency_error_ton: float = 0.0
    consistency_flags: List[str] = None
    
    # All flows for debugging
    all_flows: List[NetFlow] = None
    
    def __post_init__(self):
        if self.consistency_flags is None:
            self.consistency_flags = []
        if self.all_flows is None:
            self.all_flows = []
        if self.fee_hits is None:
            self.fee_hits = []


class ImprovedFinancialExtractor:
    """Improved financial extractor using value-flow-first approach"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # PHASE 1: State wird jetzt zentral verwaltet
        self.state = ExtractionState()
        
        # Load marketplace fee addresses from config
        from .config import (
            FRAGMENT_FEE_ADDRESS, MARKETAPP_FEE_ADDRESS, MARKETAPP_MARKETPLACE_ADDRESS,
            GETGEMS_FEE_ADDRESS, GETGEMS_SALES_ADDRESS, ALL_MARKETPLACE_ADDRESSES
        )
        
        # Create marketplace whitelist with types
        self.marketplace_addresses = {
            FRAGMENT_FEE_ADDRESS: 'fragment',
            MARKETAPP_FEE_ADDRESS: 'marketapp', 
            MARKETAPP_MARKETPLACE_ADDRESS: 'marketapp',
            GETGEMS_FEE_ADDRESS: 'getgems',
            GETGEMS_SALES_ADDRESS: 'getgems'
        }
        
        # Consistency thresholds
        self.CONSISTENCY_THRESHOLD_TON = 0.1  # 0.1 TON tolerance for rounding
        self.MIN_SIGNIFICANT_AMOUNT_TON = 0.001  # Ignore dust amounts
        
        # Fee hits tracking (reset per event)
        self.fee_hits = []
        
        # PHASE 4: Initialize UnifiedFeeCollector
        self.fee_collector = UnifiedFeeCollector(
            logger=self.logger,
            marketplace_addresses=list(self.marketplace_addresses.keys()),
            normalize_fn=self._normalize_address
        )
    
    def extract_financial_analysis(self, event: Dict, trace: Dict, nft_address: str, is_mint: bool = False) -> FinancialAnalysis:
        """
        MAIN EXTRACTION FUNCTION
        
        Strategy:
        1. Aggregate all value flows by address → Net flows
        2. Identify marketplace fee accounts from whitelist
        3. Determine seller (largest positive net flow, excluding fees)
        4. Determine buyer (largest negative net flow)
        5. Perform consistency checks
        """
        event_id = event.get('event_id', trace.get('transaction', {}).get('hash', 'unknown'))
        self.logger.debug(f"💰 IMPROVED: Starting value-flow-first extraction for NFT {nft_address[-10:]} (Event: {event_id})")
        
        # PHASE 1: State Management - Reset für neue Extraction
        self.state.reset_for_new_extraction()
        
        # PHASE 2: Index Consolidation - Build all indices once
        self.logger.debug(f"   📇 Building unified trace indices...")
        self.state.trace_indices = TraceIndices(trace, self._normalize_address)
        self.logger.debug(f"   📇 {self.state.trace_indices.stats()}")
        
        # Initialize state for current extraction
        self.state.has_seller_credits = False
        self.state.fragment_nft_payment_detected = False
        self.state.current_event_id = event_id
        self.state.current_event = event
        self.state.current_nft_address = nft_address
        
        # Use provided mint flag or detect from event
        self.state.is_mint_event = is_mint or self._detect_mint_from_event(event)
        if self.state.is_mint_event:
            self.logger.debug(f"   🎯 MINT EVENT detected - relaxed validation mode (provided={is_mint})")
        
        # Step 1: Aggregate value flows (NFT-specific)
        net_flows = self._aggregate_value_flows(event, trace, nft_address)
        
        # NEW: Ensure path flows are included in net_flows for Fragment sales
        if self.state.path_flows:
            for path_flow in self.state.path_flows:
                addr = (path_flow.get('account', {}) or {}).get('address', '')
                if addr and addr != 'teleitem_bid_inferred':
                    # Check if already in net_flows
                    existing = next((nf for nf in net_flows if nf.address == addr), None)
                    if not existing:
                        # Add as new NetFlow
                        flow_type = path_flow.get('flow_type', '')
                        amount_ton = path_flow.get('amount_nanoton', 0) / 1e9
                        
                        if amount_ton > self.MIN_SIGNIFICANT_AMOUNT_TON:
                            is_marketplace = self._classify_marketplace_by_addr(addr) != 'unknown'
                            net_flows.append(NetFlow(
                                address=addr,
                                net_amount_ton=amount_ton,
                                total_in_ton=amount_ton if amount_ton > 0 else 0,
                                total_out_ton=abs(amount_ton) if amount_ton < 0 else 0,
                                is_marketplace_fee_account=is_marketplace,
                                marketplace_type=self._classify_marketplace_by_addr(addr) if is_marketplace else '',
                                flow_types=[flow_type] if flow_type else []
                            ))
                            self.logger.debug(f"   ✅ Added path flow to net_flows: {addr[-10:]} = {amount_ton:.3f} TON ({flow_type})")
        
        self.logger.debug(f"   📊 Aggregated {len(net_flows)} net flows")
        
        # Step 2: Identify marketplace fees
        marketplace_fees = self._identify_marketplace_fees(net_flows)
        self.logger.debug(f"   🏪 Found {len(marketplace_fees)} marketplace fee flows")
        
        # Step 3: Determine seller and buyer
        seller, buyer, sale_amount = self._determine_seller_buyer(net_flows, marketplace_fees)
        
        # NEW: enforce path-constrained seller (only take seller from _last_path_flows)
        seller = self._pick_seller_from_path(seller)
        
        # If still empty, fallback to path flows
        if not seller:
            # Fallback: Largest seller_credit from path_flows
            for f in getattr(self, '_last_path_flows', []):
                if f.get('flow_type') == 'seller_credit' and f.get('amount_nanoton', 0) > 0:
                    addr = f.get('account', {}).get('address', '')
                    if addr and addr != buyer:
                        seller = addr
                        self.logger.debug(f"   🎯 SELLER fallback from path flows: {seller[-10:]}")
                        break
        
        self.logger.debug(f"   👥 Seller: {seller[-10:] if seller else 'None'}, Buyer: {buyer[-10:] if buyer else 'None'}")
        self.logger.debug(f"   💸 Sale amount: {sale_amount} TON")
        
        # Update anchors in current analysis scope if available
        current_scope = getattr(self, '_current_analysis_scope', None)
        if current_scope and current_scope.anchors:
            current_scope.anchors["buyer_wallet"] = buyer or ""
            current_scope.anchors["seller_wallet"] = seller or ""
        
        # Step 4: Calculate marketplace fees by type from net flows
        fee_breakdown = self._calculate_fee_breakdown(marketplace_fees)
        
        # Step 4a: Collect additional fee hits (dual scan: scoped + relaxed)
        scope = self.state.current_analysis_scope
        self.fee_hits = []
        self.state.seen_fee_keys = set()  # Global deduplication: (marketplace, address, lt) keys
        
        # Consolidated fee scanning: only if no fee hits from flows already exist
        existing_flow_fees = sum(1 for flow in net_flows if flow.is_marketplace_fee_account)
        
        if existing_flow_fees == 0 or len(self.state.seen_fee_keys) == 0:
            # Scoped scan: strict NFT window - only if scope exists
            if scope:
                scoped_hits_before = len(self.fee_hits)
                config = FeeCollectionConfig(
                    mode='normal',
                    start_lt=scope.start_lt,
                    end_lt=scope.end_lt,
                    allowed_nodes=scope.allowed_node_ids,
                    use_allowed_nodes=False,
                    within_scope=True
                )
                fee_hits = self.fee_collector.collect_fees(trace, config)
                self.fee_hits.extend(fee_hits)
                scoped_hits = len(self.fee_hits) - scoped_hits_before
            else:
                scoped_hits = 0
            
            # Relaxed scan: extended window for post-settlement fees - only if scope exists
            if scope:
                fee_lt_slack = 200  # ~2 blocks buffer for post-settlement fees
                relaxed_hits_before = len(self.fee_hits)
                config = FeeCollectionConfig(
                    mode='relaxed',
                    start_lt=scope.start_lt,
                    end_lt=scope.end_lt + fee_lt_slack,
                    sale_contract_addr=scope.sale_contract_addr
                )
                fee_hits = self.fee_collector.collect_fees(trace, config)
                self.fee_hits.extend(fee_hits)
                relaxed_hits = len(self.fee_hits) - relaxed_hits_before
            else:
                relaxed_hits = 0
        else:
            # Skip redundant fee scanning if flows already contain marketplace fees
            scoped_hits = 0
            relaxed_hits = 0
            self.logger.debug(f"   ⏭️  Skipping redundant fee scan: {existing_flow_fees} fees already in flows")
        
        # Collect fee hits for classification evidence only (not for amounts)
        fee_hits = self.fee_hits.copy()  # Use only the scan-based hits
        
        # IMPORTANT: Don't double-count fees! fee_breakdown already contains fees from net_flows
        # fee_hits are only used for marketplace classification, not for amount calculation
        # The amounts are already counted in the value flows during traversal
        
        # Handle out-of-scope fees for CSV consistency - ONLY when valid sale exists
        if fee_hits:
            self._handle_out_of_scope_fees(fee_breakdown, fee_hits, marketplace_fees, sale_amount, seller)
        
        self.logger.debug(f"   🏪 Fee breakdown: F={fee_breakdown.get('fragment', 0):.3f} M={fee_breakdown.get('marketapp', 0):.3f} G={fee_breakdown.get('getgems', 0):.3f}")
        self.logger.debug(f"   📊 Fee evidence: {len(fee_hits)} hits (scoped={scoped_hits}, relaxed={relaxed_hits}) - {[f'{h.marketplace_type}:{h.amount_ton:.3f}' for h in fee_hits]}")
        
        if scoped_hits == 0 and relaxed_hits > 0:
            self.logger.debug(f"   ⚠️  Fee found outside NFT scope (post-settlement): {relaxed_hits} hits")
        
        # Step 5: Perform consistency checks
        total_positive, total_negative = self._calculate_totals(net_flows)
        
        # CORRECTED buyer payment calculation: exclude sale contract and NFT from buyer detection
        scope = getattr(self, '_current_analysis_scope', None)
        excluded_addresses = {
            self._normalize_address(scope.sale_contract_addr if scope and scope.sale_contract_addr else ""),
            self._normalize_address(nft_address or "")
        }
        # Add marketplace fee addresses to exclusions
        excluded_addresses.update(self._normalize_address(f.address) for f in marketplace_fees)
        
        # Find real buyer flow (negative flow from actual buyer wallet, not sale contract)
        buyer_flow = min(
            (f for f in net_flows
             if f.net_amount_ton < -self.MIN_SIGNIFICANT_AMOUNT_TON
             and self._normalize_address(f.address) not in excluded_addresses),
            key=lambda f: f.net_amount_ton,
            default=None
        )
        
        total_fees = sum(f.net_amount_ton for f in marketplace_fees)
        
        # NFT credit as buyer payment fallback (reliable for Fragment sales)
        nft_credit = next(
            (flow.net_amount_ton for flow in net_flows
             if are_addresses_equal(flow.address, nft_address)
             and flow.net_amount_ton > self.MIN_SIGNIFICANT_AMOUNT_TON),
            0.0
        )
        
        # Debug logging for buyer detection
        self.logger.debug("   🧮 NET FLOWS: " + "; ".join(
            f"{f.address[-10:]}: net={f.net_amount_ton:.6f} in={f.total_in_ton:.6f} out={f.total_out_ton:.6f}"
            for f in net_flows
        ))
        self.logger.debug(f"   🧱 EXCLUDED (buyer detect): NFT={nft_address[-10:] if nft_address else ''}, "
                          f"SALE={scope.sale_contract_addr[-10:] if scope and scope.sale_contract_addr else ''}")
        buyer_amount = abs(buyer_flow.net_amount_ton) if buyer_flow else 0.0
        self.logger.debug(f"   💳 BUYER_FLOW: "
                          f"{buyer_flow.address[-10:] if buyer_flow else 'None'} "
                          f"amount={buyer_amount:.6f} (fallback: nft_credit={nft_credit:.6f})")
        
        if buyer_flow:
            # Use actual buyer outgoing payment from real buyer wallet
            buyer_payment = abs(buyer_flow.net_amount_ton)
            self.logger.debug(f"   💳 BUYER FLOW: {buyer_flow.address[-10:]} amount={buyer_payment:.6f} TON")
        elif nft_credit > self.MIN_SIGNIFICANT_AMOUNT_TON:
            # Fragment sale fallback: use NFT credit as buyer payment
            buyer_payment = nft_credit
            self.logger.debug(f"   🎯 Fallback BUYER PAYMENT from NFT credit: {buyer_payment:.6f} TON")
        elif sale_amount > 0 and total_fees > 0:
            # Sale-only scenario with marketplace fees: seller credit + fees
            buyer_payment = sale_amount + total_fees
            self.logger.debug(f"   🔄 Sale-only scenario: buyer payment = seller credit ({sale_amount:.3f}) + fees ({total_fees:.3f}) = {buyer_payment:.3f} TON")
        else:
            # Pure transfer or unknown: use sale amount
            buyer_payment = sale_amount
            self.logger.debug(f"   📝 Transfer scenario: buyer payment = sale amount ({sale_amount:.3f} TON)")
        
        # --- BUYER ADDRESS FALLBACK (Fragment / Sale-only) ---
        if not buyer:
            # Before consistency check: try to set the real buyer wallet
            # 1) For Fragment mint_auction: Try to extract buyer from TeleitemBidInfo notification
            buyer_fallback = None
            if self.state.is_mint_event:
                teleitem_info = FragmentDetector.find_teleitem_bid_info(trace, self.logger)
                buyer_fallback = teleitem_info.get("buyer_address") if teleitem_info["found"] else None
                if buyer_fallback:
                    self.logger.debug(f"   🎯 FRAGMENT MINT_AUCTION: Buyer from TeleitemBidInfo: {buyer_fallback[-10:]}")
            
            # 2) Fallback to topmost external wallet from Event+Trace
            if not buyer_fallback:
                buyer_fallback = self._find_topmost_external_wallet(event, trace)
            # 2) Fallback to root account of trace
            if not buyer_fallback:
                buyer_fallback = trace.get("transaction", {}).get("account", {}).get("address", "")
            
            # Guards: never use Sale-Contract, NFT or Marketplace as buyer
            scope = getattr(self, '_current_analysis_scope', None)
            exclude = set()
            if scope and scope.sale_contract_addr:
                exclude.add(self._normalize_address(scope.sale_contract_addr))
            if nft_address:
                exclude.add(self._normalize_address(nft_address))
            exclude.update(self._normalize_address(a) for a in self.marketplace_addresses.keys())
            
            if buyer_fallback and not self._address_in_list(buyer_fallback, list(self.marketplace_addresses.keys())):
                buyer = buyer_fallback
                self.logger.debug(f"   🎯 BUYER FALLBACK set to topmost wallet: {buyer[-10:]}")
                # Store detected buyer for traversal change detection
                self._detected_buyer_wallet = buyer
            else:
                self.logger.debug("   ⚠️  BUYER FALLBACK not usable (contract/nft/marketplace or empty)")
        
        # --- For Fragment events: Use TeleitemBidInfo bid amount as buyer payment ---
        teleitem_bid_amount = None
        fragment_auction_fee = None
        
        # Check for TeleitemBidInfo ONLY in Fragment-related events (mint_auction AND fragment_sale)
        # Only apply TeleitemBidInfo to Fragment marketplaces
        has_fragment_signal = any('fragment' in flow.marketplace_type.lower() for flow in marketplace_fees if hasattr(flow, 'marketplace_type'))
        teleitem_info = FragmentDetector.find_teleitem_bid_info(trace, self.logger) if (self.state.is_mint_event or has_fragment_signal) else {"found": False}
        if teleitem_info["found"]:
            buyer_payment = teleitem_info["bid_amount_ton"]
            self.state.teleitem_buyer_payment = teleitem_info["bid_amount_ton"]  # Store TeleitemBidInfo amount separately
            self.state.force_buyer_payment = True
            
            if self.state.is_mint_event:
                self.logger.debug(f"   💰 FRAGMENT MINT_AUCTION: Using TeleitemBidInfo bid as buyer payment: {buyer_payment:.6f} TON")
                
                # Extract Fragment fee from auction_fill_up amount and add to marketplace_fees
                fragment_auction_fee = self._extract_fragment_auction_fee(trace)
                if fragment_auction_fee:
                    # Create a synthetic Fragment fee flow and add to marketplace_fees list
                    from .config import FRAGMENT_FEE_ADDRESS
                    fragment_fee_flow = NetFlow(
                        address=FRAGMENT_FEE_ADDRESS,
                        net_amount_ton=fragment_auction_fee,
                        total_in_ton=fragment_auction_fee,
                        total_out_ton=0.0,
                        is_marketplace_fee_account=True,
                        marketplace_type="fragment",
                        flow_types=["fragment_auction_fee"]
                    )
                    marketplace_fees.append(fragment_fee_flow)
                    
                    # Also add to fee_hits for safety net recognition
                    fragment_fee_hit = FeeHit(
                        address=FRAGMENT_FEE_ADDRESS,
                        marketplace_type="fragment",
                        amount_ton=fragment_auction_fee,
                        amount_nanoton=int(fragment_auction_fee * 1e9),
                        lt=trace.get('transaction', {}).get('lt', 0),
                        aborted=False,
                        evidence="fragment_auction_fee"
                    )
                    fee_hits.append(fragment_fee_hit)
                    
                    self.logger.debug(f"   💰 FRAGMENT MINT_AUCTION: Added Fragment fee from auction_fill_up: {fragment_auction_fee:.6f} TON")
            else:
                self.logger.debug(f"   💰 FRAGMENT SALE: Using TeleitemBidInfo bid as buyer payment: {buyer_payment:.6f} TON")

        # --- Always calculate buyer_outgoing_abs for later use ---
        buyer_outgoing_abs = sum(
            -f.get('amount_nanoton', 0) for f in getattr(self, '_last_path_flows', [])
            if f.get('flow_type') == 'buyer_outgoing' and f.get('amount_nanoton', 0) < 0
        ) / 1e9

        # --- Fallback: enforce buyer_payment from explicit buyer_outgoing only ---
        if not teleitem_bid_amount:
            if buyer_outgoing_abs > self.MIN_SIGNIFICANT_AMOUNT_TON:
                buyer_payment = buyer_outgoing_abs
                # Set force flag to prevent later overrides
                self._force_buyer_payment = True
                preserved_payment = getattr(self, '_last_buyer_payment', 0) / 1e9
                self.logger.debug(f"   💳 BUYER_OUTGOING(raw): {buyer_outgoing_abs:.6f} TON | preserved={preserved_payment:.6f} TON")
        elif getattr(self, '_last_buyer_payment', 0) > 0 and not getattr(self, '_force_buyer_payment', False):
            buyer_payment = self._last_buyer_payment / 1e9
            self._force_buyer_payment = False
            self.logger.debug(f"   💳 BUYER PAYMENT (fallback): {buyer_payment:.6f} TON")
        # (Rest deiner Buyer-Payment Fallbacks bleibt gleich)
        
        consistency_result = self._check_consistency(net_flows, marketplace_fees, sale_amount, buyer_payment)
        
        # Clear NFT-specific fee context after use
        self._nft_specific_fee_context = None
        
        # Step 6: Merge buyer/seller from transfer signals (source of truth)
        transfer_extractor = ImprovedTransferExtractor()
        transfer_result = transfer_extractor.extract_nft_transfer_addresses(event, trace, nft_address)
        
        # Override buyer/seller with transfer signals if available (higher priority than value flows)
        # Immer Buyer/Seller aus Transfer-Signalen versuchen (Quelle der truth für Adressen)
        if transfer_result.success:
            # Owner fallback: Use transfer signals if no buyer payment was found
            has_buyer_payment = any(flow.net_amount_ton < -0.1 for flow in net_flows)
            
            # SELLER-Fallback: Transfer-Signale haben Priorität über value-flow seller
            if getattr(transfer_result, "from_address", ""):
                cand = transfer_result.from_address
                # Guards: nicht NFT/Sale/Marketplace
                guard = {
                    self._normalize_address(nft_address),
                    self._normalize_address(scope.sale_contract_addr) if scope and scope.sale_contract_addr else ""
                }
                guard.update(self._normalize_address(a) for a in self.marketplace_addresses.keys())
                if not self._address_in_list(cand, list(self.marketplace_addresses.keys())):
                    if seller != cand:
                        self.logger.debug(f"   🎯 SELLER override from transfer signals: {cand[-10:]} (was: {seller[-10:] if seller else 'None'})")
                        
                        # FRAGMENT AUCTION FIX: Check if seller receives auction_fill_up in current trace
                        is_fragment_sale = any(h.marketplace_type == 'fragment' for h in fee_hits)
                        seller_gets_auction_fillup = False
                        
                        if is_fragment_sale:
                            # Check if this address receives auction_fill_up in current trace
                            auction_fillup_info = FragmentDetector.find_auction_fillup(trace, self.logger)
                            seller_gets_auction_fillup = (auction_fillup_info["found"] and 
                                                         are_addresses_equal(auction_fillup_info["recipient_address"], cand))
                            
                        if self._is_on_nft_path(cand) or seller_gets_auction_fillup:
                            if seller_gets_auction_fillup:
                                self.logger.debug(f"   ✅ FRAGMENT: Seller {cand[-10:]} receives auction_fill_up in trace")
                            seller = cand
                        else:
                            self.logger.debug(f"   🚫 Seller {cand[-10:]} not on NFT path → ignoring cross-tree override")
                            seller = ""
            
            if transfer_result.to_address:
                original_buyer = buyer
                candidate_buyer = transfer_result.to_address
                
                # BUYER GUARDS: Prevent invalid buyers
                scope = getattr(self, '_current_analysis_scope', None)
                is_valid_buyer = True
                
                # Guard 1: Sale contract is never a buyer (EXCEPT for Fragment sales)
                if scope and scope.sale_contract_addr:
                    if are_addresses_equal(candidate_buyer, scope.sale_contract_addr):
                        # Fragment exception: Allow buyer == sale_contract if Fragment evidence exists
                        has_fragment_evidence = any(h.marketplace_type == "fragment" for h in fee_hits)
                        if has_fragment_evidence:
                            self.logger.debug(f"   ✅ BUYER GUARD: Fragment exception - allowing buyer=sale_contract: {candidate_buyer[-10:]}")
                        else:
                            self.logger.debug(f"   🚫 BUYER GUARD: Sale contract rejected as buyer: {candidate_buyer[-10:]}")
                            is_valid_buyer = False
                
                # Guard 2: NFT contract is never a buyer  
                if scope and scope.nft_address:
                    if are_addresses_equal(candidate_buyer, scope.nft_address):
                        self.logger.debug(f"   🚫 BUYER GUARD: NFT contract rejected as buyer: {candidate_buyer[-10:]}")
                        is_valid_buyer = False
                
                # Guard 3: Marketplace addresses are never buyers
                if candidate_buyer in self.marketplace_addresses:
                    self.logger.debug(f"   🚫 BUYER GUARD: Marketplace address rejected as buyer: {candidate_buyer[-10:]}")
                    is_valid_buyer = False
                
                if is_valid_buyer:
                    buyer = candidate_buyer
                    self.logger.debug(f"   🎯 BUYER from transfer signals: {original_buyer[-10:] if original_buyer else 'None'} → {buyer[-10:]}")
                    # Store detected buyer for traversal change detection
                    self._detected_buyer_wallet = buyer
                else:
                    self.logger.debug(f"   ⚠️  Transfer buyer rejected, keeping original: {original_buyer[-10:] if original_buyer else 'None'}")
            
            if transfer_result.from_address:
                # For Fragment sales: use owner fallback even without direct seller credit flow
                is_fragment_sale = any(h.marketplace_type == 'fragment' for h in fee_hits)
                seller_flow_addresses = {flow.address for flow in net_flows 
                                       if flow.net_amount_ton > 0 and flow.address}
                
                if (transfer_result.from_address in seller_flow_addresses or 
                    (is_fragment_sale and not has_buyer_payment)):  # Owner fallback for Fragment
                    
                    if not buyer or self._normalize_address(transfer_result.from_address) != self._normalize_address(buyer):
                        original_seller = seller
                        seller = transfer_result.from_address
                        reason = "seller credit flow" if transfer_result.from_address in seller_flow_addresses else "owner fallback"
                        self.logger.debug(f"   🎯 SELLER from transfer signals ({reason}): {original_seller[-10:] if original_seller else 'None'} → {seller[-10:]}")
                    else:
                        self.logger.debug("   🚫 TRANSFER SELLER GUARD: from_address equals buyer → skip seller override")
        elif transfer_result.success and not (seller and sale_amount > self.MIN_SIGNIFICANT_AMOUNT_TON):
            self.logger.debug(f"   ⚠️  Transfer signals found but no valid sale detected (seller={bool(seller)}, amount={sale_amount:.3f}) - keeping transfer classification")
        
        # CSV SAFETY NET: Zero out all fees if sale_amount is 0
        if sale_amount == 0:
            self.logger.debug(f"   🚫 CSV Safety Net: Zeroing all fees for zero sale amount")
            fee_breakdown = {'fragment': 0, 'marketapp': 0, 'getgems': 0}
            
            # FRAGMENT FIX: Restore sale amount from net_flows if we have seller credits
            actual_seller_credits = sum(
                f.net_amount_ton for f in net_flows
                if ('seller_credit' in f.flow_types or 'seller_credit_inferred' in f.flow_types)
                and f.net_amount_ton > 0
                and not self._is_marketplace_contract(f.address)
            )
            if actual_seller_credits > 0:
                sale_amount = actual_seller_credits
                self.logger.debug(f"   🔧 FRAGMENT FIX: Restored sale_amount from net_flows: {sale_amount:.3f} TON")
                # Don't zero the fees if we have a valid sale
                fee_breakdown = self._calculate_fee_breakdown(marketplace_fees)
                self.logger.debug(f"   🔧 FRAGMENT FIX: Restored fees: F={fee_breakdown.get('fragment', 0):.3f}")
            else:
                # Fragment: Try to get sale amount from seller's auction_fill_up credit
                if any(h.marketplace_type == 'fragment' for h in fee_hits) and seller:
                    auction_fillup_info = FragmentDetector.find_auction_fillup(trace, self.logger)
                    if (auction_fillup_info["found"] and 
                        are_addresses_equal(auction_fillup_info["recipient_address"], seller)):
                        seller_fillup = auction_fillup_info["credit_amount_nanoton"]
                        if seller_fillup > 0:
                            sale_amount = seller_fillup / 1e9
                            self.logger.debug(f"   💰 Fragment sale amount from seller's auction_fill_up: {sale_amount:.6f} TON")
                        # Don't zero the fees if we have a valid sale
                        fee_breakdown = self._calculate_fee_breakdown(marketplace_fees)
                
        if sale_amount == 0:
            self.logger.debug(f"   🚫 No seller credits found in net_flows to restore sale amount")
        
        # CSV EXPORT VALIDATION: Final safety checks before analysis creation
        scope = getattr(self, '_current_analysis_scope', None)
        has_fragment_evidence = any(h.marketplace_type == "fragment" for h in fee_hits)
        if buyer and scope and scope.sale_contract_addr:
            if are_addresses_equal(buyer, scope.sale_contract_addr):
                # Fragment exception: Allow buyer == sale_contract if Fragment evidence exists
                if has_fragment_evidence:
                    self.logger.debug(f"   ✅ FINAL BUYER GUARD: Fragment exception - keeping buyer=sale_contract: {buyer[-10:]}")
                else:
                    self.logger.debug(f"   🚫 FINAL BUYER GUARD: Buyer equals sale contract, trying topmost wallet fallback")
                    # Try to find a better buyer instead of just clearing
                    alt_buyer = self._find_topmost_external_wallet(event, trace)
                    if not alt_buyer:
                        alt_buyer = trace.get("transaction", {}).get("account", {}).get("address", "")
                    
                    # Validate alternative buyer
                    exclude_set = {self._normalize_address(scope.sale_contract_addr)}
                    if nft_address:
                        exclude_set.add(self._normalize_address(nft_address))
                    exclude_set.update(self._normalize_address(a) for a in self.marketplace_addresses.keys())
                    
                    if alt_buyer and self._normalize_address(alt_buyer) not in exclude_set:
                        buyer = alt_buyer
                        self.logger.debug(f"   ✅ FINAL BUYER GUARD: replaced with {buyer[-10:]}")
                    else:
                        buyer = ""
                        self.logger.debug("   ⚠️  FINAL BUYER GUARD: fallback not usable → cleared")
        
        # Ensure fees are only set when we have a valid sale OR Fragment fee evidence
        # Fragment exception: Allow fees if Fragment fee evidence exists (even with sale_amount=0)
        has_fragment_evidence = any(h.marketplace_type == "fragment" for h in fee_hits)
        if sale_amount <= 0 and not has_fragment_evidence:
            fee_breakdown = {'fragment': 0, 'marketapp': 0, 'getgems': 0}
            self.logger.debug(f"   🚫 FINAL FEE GUARD: No valid sale and no Fragment evidence, fees zeroed")
        elif sale_amount <= 0 and has_fragment_evidence:
            # RESTORE Fragment fees from fee_hits when evidence exists
            original_fees = fee_breakdown.copy()
            for hit in fee_hits:
                if hit.marketplace_type == "fragment":
                    fee_breakdown['fragment'] = hit.amount_ton
                    self.logger.debug(f"   🔧 FINAL FEE RESTORE: Fragment fee restored from evidence: {hit.amount_ton:.6f} TON")
            
            # RESTORE Fragment buyer payment if preserved but not set in buyer_payment
            preserved_buyer_payment = getattr(self, '_last_buyer_payment', 0)
            if preserved_buyer_payment > 0 and buyer_payment == 0:
                buyer_payment = preserved_buyer_payment / 1e9  # Convert from nanoTON
                self.logger.debug(f"   🔧 FINAL BUYER RESTORE: Fragment buyer payment restored: {buyer_payment:.6f} TON")
            
            self.logger.debug(f"   🎯 Fragment evidence found, fees and buyer payment restored despite sale_amount={sale_amount}")
        
        # Never override buyer payment if we have forced buyer_outgoing, but respect TeleitemBidInfo for Fragment only
        if getattr(self, '_force_buyer_payment', False):
            # Only apply TeleitemBidInfo override for Fragment events (mint, mint_auction, fragment_sale)
            teleitem_payment = getattr(self, '_teleitem_buyer_payment', None)
            is_fragment_context = (self._is_mint_event or 
                                 any('fragment' in flow.marketplace_type.lower() for flow in marketplace_fees if hasattr(flow, 'marketplace_type')))
            
            if teleitem_payment and is_fragment_context:
                buyer_payment = teleitem_payment
                self.logger.debug(f"   🔒 FORCE BUYER PAYMENT: Using TeleitemBidInfo amount: {buyer_payment:.6f} TON (Fragment context)")
            else:
                # For non-Fragment events, preserve the already calculated buyer_payment
                # Only fall back to buyer_outgoing_abs if buyer_payment is 0
                if buyer_payment > 0:
                    # Keep the existing buyer_payment that was already calculated
                    self.logger.debug(f"   🔒 FORCE BUYER PAYMENT: Preserving calculated buyer payment: {buyer_payment:.6f} TON")
                else:
                    buyer_payment = buyer_outgoing_abs
                    self.logger.debug(f"   🔒 FORCE BUYER PAYMENT: Using buyer outgoing fallback: {buyer_payment:.6f} TON")
        
        # FINAL SELLER GUARD: never allow seller == buyer
        scope = getattr(self, '_current_analysis_scope', None)
        if seller and buyer and are_addresses_equal(seller, buyer):
            self.logger.debug(f"   🚫 FINAL SELLER GUARD: seller equals buyer ({seller[-10:]}). Seeking alternative seller")

            exclude = {
                self._normalize_address(buyer),
            }
            if scope and scope.sale_contract_addr:
                exclude.add(self._normalize_address(scope.sale_contract_addr))
            if nft_address:
                exclude.add(self._normalize_address(nft_address))
            exclude.update(self._normalize_address(a) for a in self.marketplace_addresses.keys())

            # Prefer addresses with positive net flow (real seller credits), descending by amount
            alt_sellers = sorted(
                [f for f in net_flows
                 if f.net_amount_ton > self.MIN_SIGNIFICANT_AMOUNT_TON
                 and f.address
                 and self._normalize_address(f.address) not in exclude],
                key=lambda f: f.net_amount_ton,
                reverse=True
            )

            if alt_sellers:
                old = seller
                seller = alt_sellers[0].address
                self.logger.debug(f"   ✅ FINAL SELLER GUARD: replaced {old[-10:]} → {seller[-10:]}")
            else:
                self.logger.debug("   ⚠️  FINAL SELLER GUARD: no alternative seller found → clearing seller")
                seller = ""
        
        # Create final analysis
        analysis = FinancialAnalysis(
            seller_address=seller,
            buyer_address=buyer,
            sale_amount_ton=sale_amount,
            buyer_payment_ton=buyer_payment,
            fragment_fee_ton=fee_breakdown.get('fragment', 0),
            marketapp_fee_ton=fee_breakdown.get('marketapp', 0),
            getgems_fee_ton=fee_breakdown.get('getgems', 0),
            fee_hits=fee_hits,  # NEW: Raw fee evidence for marketplace classification
            scope=getattr(self, '_current_analysis_scope', None),  # NEW: Analysis scope for scoped processing
            total_positive_flow_ton=total_positive,
            total_negative_flow_ton=abs(total_negative),
            is_consistent=consistency_result['is_consistent'],
            consistency_error_ton=consistency_result['error_ton'],
            consistency_flags=consistency_result['flags'],
            all_flows=net_flows
        )
        
        self.logger.debug(f"   ✅ Analysis complete: consistent={analysis.is_consistent}")
        if not analysis.is_consistent:
            self.logger.debug(f"   ⚠️  Consistency issues: {analysis.consistency_flags}")
        
        return analysis
    
    def _find_nft_specific_traces(self, trace: Dict, nft_address: str) -> List[Dict]:
        """Find all traces that contain the specific NFT address and collect all sub-traces"""
        nft_traces = []
        
        def search_trace(trace_node, depth=0):
            if not isinstance(trace_node, dict):
                return
            
            # Convert trace to string to search for NFT
            trace_str = str(trace_node)
            if nft_address in trace_str:
                # Add this trace
                nft_traces.append(trace_node)
                self.logger.debug(f"   🎯 Found NFT trace at depth {depth}: {trace_node.get('transaction', {}).get('account', {}).get('address', 'unknown')[-20:]}")
                
                # Also add all its children and sub-children
                def collect_all_children(parent_trace):
                    if 'children' in parent_trace:
                        for child in parent_trace['children']:
                            if isinstance(child, dict):
                                nft_traces.append(child)
                                self.logger.debug(f"   📎 Added child: {child.get('transaction', {}).get('account', {}).get('address', 'unknown')[-20:]}")
                                collect_all_children(child)  # Recursive for sub-children
                
                collect_all_children(trace_node)
            
            # Continue searching in children
            if 'children' in trace_node:
                for child in trace_node['children']:
                    search_trace(child, depth + 1)
        
        search_trace(trace)
        return nft_traces
    
    def _extract_nft_specific_flows_from_trace(self, trace: Dict, nft_address: str) -> List[Dict]:
        """
        Extract NFT-specific flows using seed-based path tracing algorithm
        
        Blueprint:
        1. Seed: Find NFT item address + strong ops (nft_transfer, nft_ownership_assigned, excess)
        2. Nach oben: Trace parent path to buyer wallet (root simple_send)
        3. Nach unten: Trace child path to seller credits and marketplace fees
        4. Batch trennen: Only flows on the connected payment path for this NFT
        5. Validate: Ensure path contains ≥1 strong marker for this NFT
        """
        from .config import nanoton_to_ton, TON_PRECISION_CALCULATION
        
        flows = []
        nft_path_nodes = set()  # Track all nodes on the NFT-specific path
        
        # Build by_dest index early for seller filtering context
        self._by_dest_index = self._build_by_dest_index(trace)
        
        # Step 1: Find NFT seed nodes (strong markers)
        nft_seeds = self._find_nft_seed_nodes(trace, nft_address)
        if not nft_seeds:
            self.logger.debug(f"   🚫 No NFT seeds found for {nft_address[-10:]}")
            return flows
        
        self.logger.debug(f"   🌱 Found {len(nft_seeds)} NFT seed nodes for {nft_address[-10:]}")
        
        # Step 2: Trace upward to find the connected buyer payment
        sale_contract_addr, buyer_payment_amount = self._trace_upward_to_buyer(trace, nft_seeds, nft_address)
        if not sale_contract_addr:
            self.logger.debug(f"   ⬆️  No connected buyer payment found for {nft_address[-10:]} - trying sale-only fallback...")
            # FRAGMENT FALLBACK: Try to find sale contract from NFT seeds directly for Fragment sales
            return self._extract_sale_only_flows(trace, nft_seeds, nft_address)
        
        self.logger.debug(f"   ⬆️  Buyer payment: {buyer_payment_amount / 1e9:.6f} TON → {sale_contract_addr[-20:]}")
        
        # Store buyer payment and addresses for downstream matching
        self._last_buyer_payment = int(buyer_payment_amount)
        self._last_sale_contract_addr = sale_contract_addr
        self._current_nft_address = nft_address
        
        # Set up NFT-specific fee context for strict fee collection (now that sale_contract_addr is known)
        self._setup_nft_specific_fee_context(trace, sale_contract_addr, nft_address, 0, 999999999999)
        
        # Step 3: Trace downward from sale contract to collect all flows on this path
        path_flows, analysis_scope = self._trace_downward_from_sale_contract(trace, sale_contract_addr, nft_address)
        
        # Store path flows and analysis scope for seller validation
        self.state.path_flows = path_flows[:]
        self.state.current_analysis_scope = analysis_scope
        
        # Store sale contract address and by_dest index for strict seller filtering
        self.state.sale_contract_address = sale_contract_addr
        self.state.by_dest_index = getattr(self, '_by_dest_index', None) or self.state.by_dest_index
        
        # Debug context availability
        self.logger.debug(f"   🧱 CONTEXT: by_dest={len(self.state.by_dest_index)} sale_contract={self.state.sale_contract_address[-10:] if self.state.sale_contract_address else None}")
        
        # Step 4: Add buyer payment flow (use topmost external wallet, not trace root)
        event = self.state.current_event  # Get full event from context
        buyer_addr = self._find_topmost_external_wallet(event, trace)
        
        # CRITICAL FIX: If no buyer wallet found, use trace root as fallback to ensure buyer_outgoing flow exists
        if not buyer_addr:
            buyer_addr = trace.get("transaction", {}).get("account", {}).get("address", "")
            if buyer_addr:
                self.logger.debug(f"   👤 Using trace root as buyer fallback: {buyer_addr[-20:]}")
        
        # Always add buyer_outgoing if we have payment amount and any buyer address
        if buyer_addr and buyer_payment_amount > 0:
            # Only add buyer outgoing payment - sale contract credit will be collected in traversal
            flows.append({
                'account': {'address': buyer_addr},
                'amount_nanoton': -buyer_payment_amount,  # Negative nanoTON: money going OUT
                'flow_type': 'buyer_outgoing', 
                'evidence': f'Buyer payment: {buyer_payment_amount / 1e9:.6f} TON → {sale_contract_addr[-20:]}'
            })
            
            self.logger.debug(f"   💰 BUYER OUTGOING: {buyer_addr[-20:]} = -{buyer_payment_amount / 1e9:.6f} TON")
        
        # Step 5: Add all flows from the NFT-specific path
        flows.extend(path_flows)
        
        # Step 6: Validate the path (safety checks)
        if not self._validate_nft_path(flows, nft_address, sale_contract_addr, getattr(self, '_is_mint_event', False)):
            self.logger.debug(f"   ❌ Path validation failed for {nft_address[-10:]}")
            
            # POST-PASS: Try seller-rescue if validation failed due to missing seller credits
            remaining_budget = getattr(self, '_last_remaining_budget', {'amount': 0})
            if (remaining_budget['amount'] > max(1e8, 0.01 * max(getattr(self, '_last_buyer_payment', 1e9), 1)) and
                not getattr(self, '_has_seller_credits', False)):
                
                self.logger.debug(f"   🛟 POST-PASS: Attempting seller-rescue with {remaining_budget['amount']/1e9:.3f} TON budget")
                
                # Get a copy of allowed nodes for the post-pass
                post_allowed_nodes = getattr(self, '_last_allowed_nodes', set()).copy()
                
                # Find rescue candidate
                rescue_node_id = self._allow_top_credit_candidate(
                    trace, getattr(self, '_last_start_lt', 0), getattr(self, '_last_end_lt', 0),
                    remaining_budget, post_allowed_nodes
                )
                
                if rescue_node_id:
                    post_allowed_nodes.add(rescue_node_id)
                    self.logger.debug(f"   🛟 POST-PASS: Re-traversing with seller-rescue node")
                    
                    # Re-traverse with the rescue node
                    flows_rescue = []
                    visited_rescue = set()
                    self._traverse_outgoing_flows_restricted(
                        getattr(self, '_last_sale_contract_node', None), flows_rescue, visited_rescue, nft_address,
                        getattr(self, '_last_by_source_index', {}), post_allowed_nodes, getattr(self, '_last_start_lt', 0), 
                        getattr(self, '_last_end_lt', 0), remaining_budget, sale_contract_addr, 
                        getattr(self, '_last_nft_path_hint', set()),
                        unlimited_budget=True
                    )
                    
                    if flows_rescue:
                        self.logger.debug(f"   🛟 POST-PASS: Rescued {len(flows_rescue)} additional flows")
                        flows.extend(flows_rescue)
                        
                        # Re-validate with rescued flows
                        if self._validate_nft_path(flows, nft_address, sale_contract_addr, getattr(self, '_is_mint_event', False)):
                            self.logger.debug(f"   ✅ POST-PASS: Validation now successful!")
                            return flows
            
            # FRAGMENT FALLBACK: If validation failed but we have Fragment buyer payment, preserve existing flows
            if getattr(self, '_fragment_nft_payment_detected', False) and getattr(self, '_last_buyer_payment', 0) > 0:
                buyer_payment = getattr(self, '_last_buyer_payment', 0)
                event = getattr(self, '_current_event', {})
                buyer_addr = self._find_topmost_external_wallet(event, trace)
                
                if buyer_addr:
                    # Check if buyer_outgoing already exists in flows
                    has_buyer_flow = any(f.get('flow_type') == 'buyer_outgoing' for f in flows)
                    if not has_buyer_flow:
                        buyer_flow = {
                            'account': {'address': buyer_addr},
                            'amount_nanoton': -buyer_payment,
                            'flow_type': 'buyer_outgoing',
                            'evidence': f'Fragment fallback: buyer payment {buyer_payment / 1e9:.6f} TON'
                        }
                        flows.append(buyer_flow)
                        self.logger.debug(f"   🛟 Fragment fallback: added buyer payment {buyer_payment / 1e9:.6f} TON")
                    
                    # Preserve existing seller credits and other flows
                    self.logger.debug(f"   🛟 Fragment fallback: preserving {len(flows)} existing flows")
                    return flows  # Return merged flows, not just fallback
            
            return []  # Return empty flows if validation fails
        
        self.logger.debug(f"   🎯 NFT-specific path: {len(flows)} flows for {nft_address[-10:]} (validated)")
        return flows
    
    def _validate_nft_path(self, flows: List[Dict], nft_address: str, sale_contract_addr: str, is_mint: bool = False, relaxed: bool = False) -> bool:
        """Validate that the extracted path is correct for this NFT"""
        if not flows:
            return False
        
        # Check 1: Buyer payment (required for normal sales, optional for sale-only fallback)
        has_buyer_payment = any(
            flow.get('flow_type') in ['buyer_outgoing', 'seller_credit_inferred'] and flow.get('amount_nanoton', 0) != 0
            for flow in flows
        )
        
        # Sale-only mode: Don't require buyer payment if we have seller credits + fees
        has_seller_or_fee_flows = any(
            flow.get('flow_type') in ['seller_credit', 'marketplace_fee'] and flow.get('amount_nanoton', 0) > 0
            for flow in flows
        )
        
        if not has_buyer_payment and not relaxed and not has_seller_or_fee_flows:
            return False
        
        # Check 2: NFT involvement can be through seed-based path tracing (more lenient)
        # Accept if we have meaningful value flows that suggest NFT sale activity
        nft_normalized = self._normalize_address(nft_address)
        has_direct_nft_involvement = any(
            self._normalize_address(flow.get('account', {}).get('address', '')) == nft_normalized
            for flow in flows
        )
        
        # Alternative: If we have seller credits, assume NFT involvement through path tracing
        has_seller_credits = any(
            flow.get('flow_type') in ['seller_credit', 'marketplace_fee'] and flow.get('amount_nanoton', 0) > 0
            for flow in flows
        )
        
        # Store flag for seller-rescue logic
        self.state.has_seller_credits = has_seller_credits
        
        # NFT involvement is satisfied if either direct or through meaningful seller credits
        has_nft_involvement = has_direct_nft_involvement or has_seller_credits
        
        # Check 4: Value conservation (approximately balanced) - relative to buyer payment  
        # NEW: Capped positive sum approach to prevent budget overshoot validation errors
        included_types = {'buyer_outgoing', 'seller_credit', 'seller_credit_inferred', 'marketplace_fee', 'gas_fee'}
        
        buyer_out = -sum(f.get('amount_nanoton', 0) for f in flows
                        if f.get('flow_type') == 'buyer_outgoing')
        
        pos_items = [f.get('amount_nanoton', 0) for f in flows
                    if f.get('flow_type') in ('seller_credit', 'seller_credit_inferred', 'marketplace_fee') 
                    and f.get('flow_type') != 'buyer_change'  # Explicitly exclude buyer change
                    and f.get('amount_nanoton', 0) > 0]
        
        # Laufend bis zur Buyer-Zahlung aufsummieren (Kappung)
        remaining = buyer_out
        capped_pos_sum = 0
        for a in pos_items:
            take = min(remaining, a)
            capped_pos_sum += take
            remaining -= take
            if remaining <= 0:
                break
        
        residual = capped_pos_sum - buyer_out  # sollte ~0 sein
        buyer_payment_abs = abs(next(
            (f.get('amount_nanoton', 0) for f in flows if f.get('flow_type') == 'buyer_outgoing'),
            0
        ))
        
        # Fallback for sale-only scenarios: use seller credit or stored buyer payment
        if buyer_payment_abs == 0:
            seller_credit_abs = abs(max(
                (f.get('amount_nanoton', 0) for f in flows 
                 if f.get('flow_type') in ('seller_credit', 'seller_credit_inferred')),
                default=0
            ))
            buyer_payment_abs = seller_credit_abs or abs(getattr(self, '_last_buyer_payment', 0))
        
        # FRAGMENT FIX: Include relaxed Fragment fees in balance calculation
        if getattr(self, '_fragment_nft_payment_detected', False) and hasattr(self, 'fee_hits'):
            residual_abs = abs(residual)
            relaxed_fees = [h.amount_nanoton for h in self.fee_hits 
                          if h.marketplace_type == 'fragment' and not h.within_scope]
            if relaxed_fees:
                best_fee = min(relaxed_fees, key=lambda a: abs(residual_abs - a))
                fee_gap = abs(residual_abs - best_fee)
                if fee_gap <= max(1e8, 0.005 * max(buyer_payment_abs, best_fee)):
                    self.logger.debug(f"   💰 Fragment balance: residual={residual_abs/1e9:.6f} TON, fee={best_fee/1e9:.6f} TON, gap={fee_gap/1e9:.6f} TON → balanced")
                    residual += best_fee  # Add virtual fee flow for balance check
        
        # Base tolerance: 0.1 TON oder 0.5% der Buyer-Zahlung
        tolerance = max(1e8, 0.005 * buyer_out)  # 0.1 TON oder 0.5 %
        is_approximately_balanced = abs(residual) <= tolerance
        
        # For mint events, seller credits are not required (buyer pays directly to marketplace)
        if is_mint:
            validation_passed = has_buyer_payment and has_nft_involvement and is_approximately_balanced
        else:
            # Enhanced sale-only validation: balanced + (seller credits OR marketplace fees) + NFT involvement
            sale_only_valid = (
                has_seller_or_fee_flows and 
                has_nft_involvement and 
                is_approximately_balanced
            )
            normal_sale_valid = (
                has_buyer_payment and 
                has_nft_involvement and 
                has_seller_credits and 
                is_approximately_balanced
            )
            
            # FRAGMENT VALIDATOR: Accept Fragment sales with buyer payment + NFT + balanced flows
            fragment_valid = (getattr(self, '_fragment_nft_payment_detected', False) and 
                            has_buyer_payment and has_nft_involvement and is_approximately_balanced)
            
            if fragment_valid:
                self.logger.debug(f"   🎯 Fragment validator: buyer payment + NFT involvement + balanced flows")
            
            # Accept either normal sale, balanced sale-only, or Fragment scenario
            validation_passed = normal_sale_valid or sale_only_valid or fragment_valid
        
        if not validation_passed:
            self.logger.debug(f"   ⚠️  Path validation details (is_mint={is_mint}):")
            self.logger.debug(f"      Buyer payment: {has_buyer_payment}")
            self.logger.debug(f"      NFT involvement: {has_nft_involvement} (direct: {has_direct_nft_involvement})")
            self.logger.debug(f"      Seller credits: {has_seller_credits}")
            self.logger.debug(f"      Balanced: {is_approximately_balanced} (residual: {residual/1e9:.6f} TON, +capped={capped_pos_sum/1e9:.6f} / buyer={buyer_out/1e9:.6f})")
        else:
            self.logger.debug(f"   ✅ Path validation passed (is_mint={is_mint})")
        
        return validation_passed
    
    def _detect_mint_from_event(self, event: Dict) -> bool:
        """Detect if this is a mint event based on event actions"""
        actions = event.get('actions', [])
        self.logger.debug(f"   🔍 MINT CHECK: Found {len(actions)} actions: {[a.get('type', 'unknown') for a in actions]}")
        for action in actions:
            action_type = action.get('type', '')
            if action_type == 'NftItemMint':
                self.logger.debug(f"   ✅ MINT: Found NftItemMint action")
                return True
            # TeleitemBidInfo removed - Fragment sales should not be treated as mints
        self.logger.debug(f"   ❌ MINT: No mint actions found")
        return False
    
    def _find_nft_seed_nodes(self, trace: Dict, nft_address: str) -> List[Dict]:
        """Find all nodes that contain strong markers for this NFT"""
        seeds = []
        nft_normalized = self._normalize_address(nft_address)
        
        def search_node(node):
            if not isinstance(node, dict):
                return
            
            # Check if transaction involves our NFT
            tx = node.get('transaction', {})
            if not tx:
                return
            
            account_addr = tx.get('account', {}).get('address', '')
            account_normalized = self._normalize_address(account_addr)
            
            # Method 1: Account address exactly matches NFT (primary seed)
            if account_normalized == nft_normalized:
                seeds.append(node)
                return
            
            # Method 2: Strong ops but ONLY if NFT address is found in the message body/raw_body
            strong_ops = ['0x5fcc3d14', '0x05138d91', '0xd53276db']  # nft_transfer, nft_ownership_assigned, excess
            
            # Check out_msgs for strong ops + NFT address in body
            if 'out_msgs' in tx:
                for msg in tx['out_msgs']:
                    if isinstance(msg, dict) and msg.get('op_code') in strong_ops:
                        # Only add as seed if NFT address appears in the message body
                        if self._nft_address_in_message_body(msg, nft_address):
                            seeds.append(node)
                            return
            
            # Check in_msg for strong ops + NFT address in body
            if 'in_msg' in tx:
                in_msg = tx['in_msg']
                if isinstance(in_msg, dict) and in_msg.get('op_code') in strong_ops:
                    # Only add as seed if NFT address appears in the message body
                    if self._nft_address_in_message_body(in_msg, nft_address):
                        seeds.append(node)
                    return
        
        # Search entire trace tree
        UnifiedTraversal.simple_traverse(trace, search_node)
        return seeds
    
    def _nft_address_in_message_body(self, message: Dict, nft_address: str) -> bool:
        """Check if NFT address appears in message body or raw_body"""
        if not message or not nft_address:
            return False
        
        nft_normalized = self._normalize_address(nft_address)
        nft_short = nft_address[-20:].lower()  # Last 20 chars for partial matching
        
        # Check decoded_body first
        decoded_body = message.get('decoded_body', {})
        if decoded_body and isinstance(decoded_body, dict):
            # Convert entire decoded body to string and search
            body_str = str(decoded_body).lower()
            if nft_short in body_str or nft_normalized.lower() in body_str:
                return True
        
        # Check raw_body as fallback
        raw_body = message.get('raw_body', '')
        if raw_body and isinstance(raw_body, str):
            # Search in raw body (base64 or hex)
            raw_lower = raw_body.lower()
            if nft_short in raw_lower or nft_normalized.lower() in raw_lower:
                return True
        
        return False
    
    def _msg_references_nft(self, msg: Dict, nft_address: str) -> bool:
        """Check if message contains strong ops referencing our NFT"""
        if not isinstance(msg, dict):
            return False
        
        # Check op codes for strong markers
        op_code = msg.get('op_code', '')
        strong_ops = ['0x5fcc3d14', '0x05138d91', '0xd53276db']  # nft_transfer, nft_ownership_assigned, excess
        
        if op_code in strong_ops:
            # For strong ops, be more lenient - don't require NFT address in body
            # The op code itself is a strong signal of NFT involvement
            return True
        
        return False
    
    def _trace_upward_to_buyer(self, trace: Dict, nft_seeds: List[Dict], nft_address: str) -> tuple:
        """Trace upward from NFT seeds to find connected buyer payment"""
        # Find the sale contract that sent money to our NFT
        sale_contract_candidates = set()
        
        self.logger.debug(f"   🔍 Examining {len(nft_seeds)} NFT seeds for {nft_address[-10:]}")
        
        for i, seed in enumerate(nft_seeds):
            tx = seed.get('transaction', {})
            account_addr = tx.get('account', {}).get('address', '')
            self.logger.debug(f"   🌱 Seed {i+1}: {account_addr[-20:] if account_addr else 'no-addr'}")
            
            if 'in_msg' in tx:
                source_addr = tx['in_msg'].get('source', {}).get('address', '')
                if source_addr:
                    sale_contract_candidates.add(source_addr)
                    self.logger.debug(f"   ⬆️  Found source: {source_addr[-20:]} → {account_addr[-20:]}")
                else:
                    self.logger.debug(f"   ⬆️  No source in in_msg for {account_addr[-20:]}")
            else:
                self.logger.debug(f"   ⬆️  No in_msg for {account_addr[-20:]}")
        
        self.logger.debug(f"   📝 Sale contract candidates: {len(sale_contract_candidates)}")
        for addr in sale_contract_candidates:
            self.logger.debug(f"   🏪 Candidate: {addr[-20:]}")
        
        if not sale_contract_candidates:
            # 1) ausgehend von Knoten, die an den NFT senden (by-dest)
            by_dest = self._build_by_dest_index(trace)
            self.logger.debug(f"   🔄 No sale contract candidates from in_msg.source, trying by-dest approach")
            
            # Find nodes that send messages to our NFT seeds
            for seed in nft_seeds:
                tx = seed.get('transaction', {})
                account_addr = tx.get('account', {}).get('address', '')
                if account_addr in by_dest:
                    senders = by_dest[account_addr]
                    self.logger.debug(f"   📨 Found {len(senders)} senders to NFT {account_addr[-20:]}: {[s[-20:] for s in senders]}")
                    sale_contract_candidates.update(senders)
            
            # If still no candidates, expand to all non-NFT nodes
            if not sale_contract_candidates:
                self.logger.debug(f"   🔄 Still no candidates, expanding to all non-NFT nodes")
                all_nodes = self._extract_all_nodes(trace)
                for node_addr in all_nodes:
                    if not are_addresses_equal(node_addr, nft_address):
                        sale_contract_candidates.add(node_addr)
                
                self.logger.debug(f"   📝 Expanded to {len(sale_contract_candidates)} total candidates")
            
            if not sale_contract_candidates:
                return None, 0
        
        # Check root transaction for buyer payments to sale contracts
        root_tx = trace.get('transaction', {})
        if 'in_msg' not in root_tx or 'decoded_body' not in root_tx['in_msg']:
            return None, 0
        
        decoded = root_tx['in_msg']['decoded_body']
        
        # Helper function to check payment and return result if match found
        def _check_payment_match(dest_addr, value_info, source_desc):
            if not dest_addr or not isinstance(value_info, dict):
                return None
            
            dest_normalized = self._normalize_address(dest_addr)
            matched_candidate = None
            marketplace_candidates = []
            other_candidates = []
            
            # Separate marketplace vs other candidates - include NFT candidates for Fragment sales
            all_candidates = list(sale_contract_candidates) + ([nft_address] if nft_address else [])
            for candidate in all_candidates:
                candidate_normalized = self._normalize_address(candidate)
                if dest_normalized == candidate_normalized:
                    if self._is_marketplace_contract(candidate):
                        marketplace_candidates.append(candidate)
                    else:
                        other_candidates.append(candidate)
            
            # Prioritize marketplace contracts
            if marketplace_candidates:
                matched_candidate = marketplace_candidates[0]
                self.logger.debug(f"   🏪 {source_desc} MARKETPLACE MATCH: {matched_candidate[-20:]}")
            elif other_candidates:
                matched_candidate = other_candidates[0]
            
            if matched_candidate:
                try:
                    value_nanoton = int(value_info['grams'])
                    from .config import nanoton_to_ton, TON_PRECISION_CALCULATION
                    value_ton = nanoton_to_ton(value_nanoton, TON_PRECISION_CALCULATION)
                    # For Fragment sales: if payment goes to NFT contract, don't change sale contract
                    if dest_addr == nft_address:
                        self.logger.debug(f"   ✅ {source_desc} BUYER PAYMENT (NFT): {value_ton:.6f} TON → {dest_addr[-20:]} (Fragment-style)")
                        # Set flag and preserve buyer payment for Fragment-style LT window extension
                        self.state.fragment_nft_payment_detected = True
                        self.state.buyer_payment_nanoton = int(value_nanoton)  # Preserve buyer payment before validation
                        self.logger.debug(f"   🎯 Preserved buyer payment for Fragment analysis: {value_ton:.6f} TON")
                        # Return first sale contract candidate to preserve original sale contract
                        original_sale_contract = next(iter(sale_contract_candidates)) if sale_contract_candidates else dest_addr
                        return original_sale_contract, value_nanoton
                    else:
                        self.logger.debug(f"   ✅ {source_desc} BUYER PAYMENT: {value_ton:.6f} TON → {dest_addr[-20:]}")
                        return dest_addr, value_nanoton
                except (ValueError, TypeError, KeyError) as e:
                    self.logger.debug(f"   ❌ {source_desc}: Value parsing error - {e}")
            return None
        
        # 1. Traditional simple_send (v3/v4 wallets) 
        self.logger.debug(f"   🔍 SIMPLE_SEND DEBUG: decoded keys = {list(decoded.keys()) if decoded else 'None'}")
        if 'payload' in decoded:
            payload = decoded['payload']
            if isinstance(payload, dict):
                self.logger.debug(f"   🔍 SIMPLE_SEND DEBUG: payload keys = {list(payload.keys())}")
            else:
                self.logger.debug(f"   🔍 SIMPLE_SEND DEBUG: payload type = {type(payload)} (not dict)")
        
        if 'payload' in decoded and isinstance(decoded['payload'], dict) and 'simple_send' in decoded['payload']:
            simple_send = decoded['payload']['simple_send']
            self.logger.debug(f"   🔍 SIMPLE_SEND DEBUG: simple_send keys = {list(simple_send.keys()) if isinstance(simple_send, dict) else f'type={type(simple_send)}'}")
            if 'payload' in simple_send:
                self.logger.debug(f"   💰 Checking simple_send payments...")
                # For Fragment sales, also check payments to NFT contract
                nft_candidates = [nft_address] if nft_address else []
                all_candidates = list(sale_contract_candidates) + nft_candidates
                self.logger.debug(f"   🔍 SIMPLE_SEND: Checking {len(all_candidates)} candidates: {[c[-10:] for c in all_candidates]}")
                
                for i, entry in enumerate(simple_send['payload']):
                    if not isinstance(entry, dict) or 'message' not in entry:
                        continue
                    
                    message = entry['message']
                    if not isinstance(message, dict) or 'message_internal' not in message:
                        continue
                    
                    msg_internal = message['message_internal']
                    if not isinstance(msg_internal, dict):
                        continue
                    
                    result = _check_payment_match(
                        msg_internal.get('dest', ''), 
                        msg_internal.get('value', {}),
                        f"simple_send[{i+1}]"
                    )
                    if result:
                        return result
        
        # 2. Wallet v5 signed external transactions
        decoded_op_name = root_tx['in_msg'].get('decoded_op_name', '')
        if decoded_op_name == 'wallet_signed_external_v5_r1':
            self.logger.debug(f"   💰 Checking wallet_signed_external_v5_r1 payments...")
            actions = decoded.get('actions', [])
            if isinstance(actions, list):
                for i, action in enumerate(actions):
                    if not isinstance(action, dict) or 'msg' not in action:
                        continue
                    
                    msg = action['msg']
                    if not isinstance(msg, dict) or 'message_internal' not in msg:
                        continue
                    
                    msg_internal = msg['message_internal']
                    if not isinstance(msg_internal, dict):
                        continue
                    
                    result = _check_payment_match(
                        msg_internal.get('dest', ''), 
                        msg_internal.get('value', {}),
                        f"v5_actions[{i+1}]"
                    )
                    if result:
                        return result
        
        # 3. Alternative payload structures (decoded_body.payload.messages[*])
        if 'payload' in decoded and 'messages' in decoded['payload']:
            self.logger.debug(f"   💰 Checking payload.messages...")
            messages = decoded['payload']['messages']
            if isinstance(messages, list):
                for i, message in enumerate(messages):
                    if not isinstance(message, dict):
                        continue
                    
                    if 'message_internal' in message:
                        msg_internal = message['message_internal']
                        if isinstance(msg_internal, dict):
                            result = _check_payment_match(
                                msg_internal.get('dest', ''), 
                                msg_internal.get('value', {}),
                                f"payload.messages[{i+1}]"
                            )
                            if result:
                                return result
                    elif 'message' in message and isinstance(message['message'], dict):
                        if 'message_internal' in message['message']:
                            msg_internal = message['message']['message_internal']
                            if isinstance(msg_internal, dict):
                                result = _check_payment_match(
                                    msg_internal.get('dest', ''), 
                                    msg_internal.get('value', {}),
                                    f"payload.messages[{i+1}].message"
                                )
                                if result:
                                    return result
        
        # 4. Direct messages structure (decoded_body.messages[*])
        if 'messages' in decoded:
            self.logger.debug(f"   💰 Checking direct messages...")
            messages = decoded['messages']
            if isinstance(messages, list):
                for i, message in enumerate(messages):
                    if not isinstance(message, dict):
                        continue
                    
                    if 'message_internal' in message:
                        msg_internal = message['message_internal']
                        if isinstance(msg_internal, dict):
                            result = _check_payment_match(
                                msg_internal.get('dest', ''), 
                                msg_internal.get('value', {}),
                                f"messages[{i+1}]"
                            )
                            if result:
                                return result
        
        # 5. Root out_msgs fallback (scan transaction outputs directly)
        self.logger.debug(f"   💰 Checking root out_msgs fallback...")
        out_msgs = root_tx.get('out_msgs', [])
        if not isinstance(out_msgs, list):
            out_msgs = [out_msgs] if out_msgs else []
        
        for i, msg in enumerate(out_msgs):
            if not isinstance(msg, dict):
                continue
            if msg.get('msg_type') != 'int_msg':
                continue
            
            # Extract destination from multiple possible fields
            dest_addr = ''
            if 'destination' in msg and isinstance(msg['destination'], dict):
                dest_addr = msg['destination'].get('address', '')
            elif 'dest' in msg:
                dest_addr = msg.get('dest', '')
            
            if not dest_addr:
                continue
            
            # Check if destination matches any sale contract
            dest_normalized = self._normalize_address(dest_addr)
            matched_candidate = None
            for candidate in sale_contract_candidates:
                if dest_normalized == self._normalize_address(candidate):
                    matched_candidate = candidate
                    break
            
            if matched_candidate:
                # Extract value from multiple possible structures
                value_nanoton = 0
                if 'value' in msg:
                    value_info = msg['value']
                    if isinstance(value_info, dict) and 'grams' in value_info:
                        try:
                            value_nanoton = int(value_info['grams'])
                        except (ValueError, TypeError):
                            continue
                    elif isinstance(value_info, (int, str)):
                        try:
                            value_nanoton = int(value_info)
                        except (ValueError, TypeError):
                            continue
                
                if value_nanoton > 0:
                    from .config import nanoton_to_ton, TON_PRECISION_CALCULATION
                    value_ton = nanoton_to_ton(value_nanoton, TON_PRECISION_CALCULATION)
                    marketplace_flag = "🏪 " if self._is_marketplace_contract(matched_candidate) else ""
                    self.logger.debug(f"   ✅ {marketplace_flag}out_msgs[{i+1}] BUYER PAYMENT: {value_ton:.6f} TON → {dest_addr[-20:]}")
                    return dest_addr, value_nanoton
        
        # FRAGMENT FALLBACK: If no payment found via standard methods, try Fragment-specific approaches
        self.logger.debug(f"   🔍 No buyer payment found via standard methods - trying Fragment fallbacks...")
        
        # A) Scan for auction_fill_up operations in trace children  
        auction_fillup_info = FragmentDetector.find_auction_fillup(trace, self.logger)
        if auction_fillup_info["found"]:
            # Check if this connects to any of our sale contract candidates
            for candidate in sale_contract_candidates:
                if self._is_connected_to_sale_contract({'transaction': {'account': {'address': auction_fillup_info["recipient_address"]}}}, candidate, trace):
                    self.logger.debug(f"   ✅ auction_fill_up connected to sale contract {candidate[-20:]}")
                    return candidate, auction_fillup_info["credit_amount_nanoton"]
        
        # B) Use TeleitemBidInfo as price source if available
        teleitem_info = FragmentDetector.find_teleitem_bid_info(trace, self.logger)
        if teleitem_info["found"]:
            # Find Fragment auction config beneficiary as sale contract
            fragment_beneficiary = self._find_fragment_beneficiary(trace, nft_address) 
            if fragment_beneficiary:
                self.logger.debug(f"   💰 Using TeleitemBidInfo amount {teleitem_info['bid_amount_ton']:.6f} TON with Fragment beneficiary {fragment_beneficiary[-20:]}")
                return fragment_beneficiary, teleitem_info["bid_amount_nanoton"]
            # Fallback to first sale contract candidate
            elif sale_contract_candidates:
                first_candidate = list(sale_contract_candidates)[0]
                self.logger.debug(f"   💰 Using TeleitemBidInfo amount {teleitem_info['bid_amount_ton']:.6f} TON with sale candidate {first_candidate[-20:]}")
                return first_candidate, teleitem_info["bid_amount_nanoton"]
        
        return None, 0
    
    
    def _find_fragment_beneficiary(self, trace: Dict, nft_address: str) -> str:
        """Find Fragment beneficiary address from auction_config in NFT state"""
        self.logger.debug(f"   🔍 Looking for Fragment beneficiary in auction_config...")
        
        def scan_node(node):
            if not isinstance(node, dict):
                return None
            
            tx = node.get('transaction', {})
            if not tx:
                return None
            
            # Check if this is the NFT address we're looking for
            account_addr = tx.get('account', {}).get('address', '')
            if not are_addresses_equal(account_addr, nft_address):
                return None
            
            # Look for auction_config in account state
            account_state = tx.get('account_state_after', {}) or tx.get('account_state_before', {})
            if isinstance(account_state, dict) and 'account' in account_state:
                account = account_state['account']
                if isinstance(account, dict) and 'storage' in account:
                    storage = account['storage']
                    if isinstance(storage, dict) and 'state' in storage:
                        state = storage['state']
                        if isinstance(state, dict) and 'auction_config' in state:
                            auction_config = state['auction_config']
                            if isinstance(auction_config, dict) and 'beneficiar_address' in auction_config:
                                beneficiary = auction_config['beneficiar_address']
                                if beneficiary:
                                    self.logger.debug(f"   🎯 Found Fragment beneficiary: {beneficiary[-20:]}")
                                    return beneficiary
            
            return None
        
        result = UnifiedTraversal.find_first(trace, scan_node)
        return result if result else None
    
    def _is_connected_to_sale_contract(self, node: Dict, sale_contract_addr: str, trace: Dict) -> bool:
        """Check if node is connected to sale contract within 1-2 hops"""
        # Simple implementation: check if node sends money to sale contract in out_msgs
        tx = node.get('transaction', {})
        if not tx:
            return False
        
        out_msgs = tx.get('out_msgs', [])
        if not isinstance(out_msgs, list):
            out_msgs = [out_msgs] if out_msgs else []
        
        sale_normalized = self._normalize_address(sale_contract_addr)
        
        for msg in out_msgs:
            if isinstance(msg, dict):
                dest_addr = msg.get('destination', {}).get('address', '') or msg.get('dest', '')
                if dest_addr and self._normalize_address(dest_addr) == sale_normalized:
                    return True
        
        return False
    
    def _extract_sale_only_flows(self, trace: Dict, nft_seeds: List[Dict], nft_address: str) -> List[Dict]:
        """Extract flows from Fragment sales without buyer payment (sale-only fallback)"""
        flows = []
        
        # Step 1: Find sale contract from NFT seeds (source addresses)
        sale_contract_candidates = set()
        for seed in nft_seeds:
            tx = seed.get('transaction', {})
            if 'in_msg' in tx:
                source_addr = tx['in_msg'].get('source', {}).get('address', '')
                if source_addr:
                    sale_contract_candidates.add(source_addr)
        
        if not sale_contract_candidates:
            self.logger.debug(f"   ❌ No sale contract candidates found in NFT seeds")
            
            # FRAGMENT CHILDREN FIX: Check if NFT seed has direct children with credit flows
            children_with_credit = []
            for seed in nft_seeds:
                if 'children' in seed:
                    for child in seed['children']:
                        child_tx = child.get('transaction', {})
                        if child_tx:
                            credit = child_tx.get('credit_phase', {}).get('credit', 0)
                            if credit > 0:
                                children_with_credit.append((child, credit))
                                
            if children_with_credit:
                self.logger.debug(f"   🔗 FRAGMENT FIX: Found {len(children_with_credit)} children with credit flows")
                for child, credit in children_with_credit:
                    child_addr = child.get('transaction', {}).get('account', {}).get('address', '')
                    marketplace_type = self._classify_marketplace_by_addr(child_addr)
                    flow_type = f"{marketplace_type}_fee" if marketplace_type != "unknown" else "seller_credit"
                    
                    self.logger.debug(f"   💰 FRAGMENT CHILD: {child_addr[-20:] if child_addr else 'unknown'} = {credit / 1e9:.6f} TON ({flow_type})")
                    
                    # Return raw flow dictionary for aggregation, not NetFlow object
                    flow_dict = {
                        'account': {'address': child_addr},
                        'amount_nanoton': credit,  # Keep in nanoTON for consistency
                        'flow_type': flow_type,
                        'evidence': f'Fragment child credit: {credit / 1e9:.6f} TON'
                    }
                    flows.append(flow_dict)
                
                # KRITISCH: Speichere flows auch in _last_path_flows für seller detection  
                self._last_path_flows = flows[:]
                
                # Debug: Verify what we're storing
                seller_credits_in_path = [f for f in self._last_path_flows 
                                        if f.get('flow_type') == 'seller_credit']
                if seller_credits_in_path:
                    self.logger.debug(f"   ✅ FRAGMENT: Stored {len(seller_credits_in_path)} seller_credit flows in _last_path_flows")
                    for sc in seller_credits_in_path:
                        addr = sc.get('account', {}).get('address', 'unknown')
                        amt = sc.get('amount_nanoton', 0) / 1e9
                        self.logger.debug(f"      -> {addr[-20:] if addr else 'unknown'}: {amt:.6f} TON")
                
                self.logger.debug(f"   ✅ FRAGMENT: Extracted {len(flows)} flows from children AND stored in _last_path_flows")
                return flows
            
            self.logger.debug(f"   🔄 No children with credit flows, falling back to broad fee window scan")
            
            # Use broad fee window when no sale contract found
            seed_lts = [seed.get('transaction', {}).get('lt', 0) for seed in nft_seeds if seed.get('transaction', {}).get('lt', 0)]
            if seed_lts:
                start_lt = min(seed_lts) - 100  # Much broader buffer before
                end_lt = max(seed_lts) + 200    # Much broader buffer after
            else:
                start_lt, end_lt = 0, 999999999999
                
            self.logger.debug(f"   ⏰ Broad fee window: {start_lt} → {end_lt}")
            
            # Collect fees with relaxed window
            config = FeeCollectionConfig(
                mode='relaxed',
                start_lt=start_lt,
                end_lt=end_lt,
                sale_contract_addr=""
            )
            fee_hits = self.fee_collector.collect_fees(trace, config)
            
            # Convert fee hits to flows
            for hit in fee_hits:
                marketplace_type = self._classify_marketplace_by_addr(hit.address)
                net_flow = NetFlow(
                    address=hit.address,
                    net_amount_ton=hit.amount_ton,
                    total_in_ton=hit.amount_ton if hit.amount_ton > 0 else 0.0,
                    total_out_ton=0.0 if hit.amount_ton > 0 else abs(hit.amount_ton),
                    is_marketplace_fee_account=True,
                    marketplace_type=marketplace_type,
                    flow_types=[f"{marketplace_type}_fee"]
                )
                flows.append(net_flow)
                
            self.logger.debug(f"   💰 Broad scan found {len(fee_hits)} fee hits")
            return flows
        
        # Step 2: Try to find Fragment beneficiary as authoritative sale contract
        fragment_beneficiary = self._find_fragment_beneficiary(trace, nft_address)
        sale_contract_addr = None
        
        if fragment_beneficiary:
            # Prefer Fragment beneficiary if it matches any candidate
            fragment_normalized = self._normalize_address(fragment_beneficiary)
            for candidate in sale_contract_candidates:
                if self._normalize_address(candidate) == fragment_normalized:
                    sale_contract_addr = fragment_beneficiary
                    self.logger.debug(f"   🎯 Using Fragment beneficiary as sale contract: {sale_contract_addr[-20:]}")
                    break
        
        # Fallback: Score candidates instead of using first
        if not sale_contract_addr:
            sale_contract_addr = self._score_sale_contract_candidates(trace, sale_contract_candidates, nft_seeds, nft_address)
            if sale_contract_addr:
                self.logger.debug(f"   🏪 Using scored candidate as sale contract: {sale_contract_addr[-20:]}")
            else:
                # Ultimate fallback to first if scoring fails
                sale_contract_addr = list(sale_contract_candidates)[0]
                self.logger.debug(f"   🏪 Scoring failed, using first candidate: {sale_contract_addr[-20:]}")
        
        # Step 3: Set up NFT-specific fee context for strict fee collection
        # Use expanded LT window similar to seller-rescue logic
        seed_lts = [seed.get('transaction', {}).get('lt', 0) for seed in nft_seeds if seed.get('transaction', {}).get('lt', 0)]
        if seed_lts:
            start_lt = min(seed_lts) - 10  # Small buffer before
            end_lt = max(seed_lts) + 50    # Larger buffer after for Fragment processing
        else:
            start_lt, end_lt = 0, 999999999999
        
        self.logger.debug(f"   ⏰ Sale-only LT window: {start_lt} → {end_lt}")
        
        # Store for fee context
        self._last_buyer_payment = 0  # No buyer payment in this fallback
        self._setup_nft_specific_fee_context(trace, sale_contract_addr, nft_address, start_lt, end_lt)
        
        # Step 4: Trace downward from sale contract to collect flows with expanded window and unlimited budget
        path_flows, analysis_scope = self._trace_downward_from_sale_contract(
            trace, sale_contract_addr, nft_address,
            override_window=(start_lt, end_lt),
            unlimited_budget=True
        )
        
        # Store path flows and analysis scope for seller validation
        self._last_path_flows = path_flows[:]
        self._current_analysis_scope = analysis_scope
        
        # Store sale contract address and by_dest index for strict seller filtering
        self._last_sale_contract_address = sale_contract_addr
        self._last_by_dest_index = getattr(self, '_by_dest_index', None) or getattr(self, '_last_by_dest_index', {})
        
        # Debug context availability
        self.logger.debug(f"   🧱 CONTEXT (fallback): by_dest={len(self._last_by_dest_index)} sale_contract={self._last_sale_contract_address[-10:] if self._last_sale_contract_address else None}")
        
        flows.extend(path_flows)
        
        # Step 5: Try to extract TeleitemBidInfo price as sale amount if no flows found
        if not flows:
            teleitem_info = FragmentDetector.find_teleitem_bid_info(trace, self.logger)
            if teleitem_info["found"]:
                bid_amount = teleitem_info["bid_amount_nanoton"]
                # Create synthetic seller credit flow using TeleitemBidInfo amount
                flows.append({
                    'account': {'address': 'teleitem_bid_inferred'},
                    'amount_nanoton': bid_amount,
                    'flow_type': 'seller_credit_inferred',
                    'evidence': f'Inferred from TeleitemBidInfo: {teleitem_info["bid_amount_ton"]:.6f} TON'
                })
                self.logger.debug(f"   💰 INFERRED SALE AMOUNT: {bid_amount / 1e9:.6f} TON from TeleitemBidInfo")
        
        # Step 6: Validate the path but with relaxed requirements for sale-only
        if not self._validate_nft_path(flows, nft_address, sale_contract_addr, False, relaxed=True):
            self.logger.debug(f"   ⚠️  Sale-only path validation failed (relaxed) for {nft_address[-10:]}")
        
        self.logger.debug(f"   ⬇️  Sale-only flows collected: {len(flows)}")
        return flows
    
    def _trace_downward_from_sale_contract(self, trace: Dict, sale_contract_addr: str, nft_address: str, 
                                         override_window: tuple = None, unlimited_budget: bool = False) -> tuple:
        """Trace downward from sale contract following out_msg flows with NFT-specific path restriction"""
        flows = []
        visited_nodes = set()
        
        # Build both indices for comprehensive routing coverage
        by_source_index = self._build_by_source_index(trace)
        by_dest_index = self._build_by_dest_index(trace)
        
        # Store both indices for later use in traversal
        self._by_source_index = by_source_index
        self._by_dest_index = by_dest_index
        
        # Find NFT seeds for path intersection
        nft_seeds = self._find_nft_seed_nodes(trace, nft_address)
        
        # A) Precompute ancestors of NFT seeds (upstream paths)
        ancestors_of_seeds = self._precompute_allowed_nodes(trace, nft_seeds)
        self.logger.debug(f"   🎯 Seed-ancestors: {len(ancestors_of_seeds)}")
        
        # B) Find the sale contract node in the trace (pass buyer_payment for better matching)
        buyer_payment_amount = getattr(self, '_last_buyer_payment', 0)
        sale_contract_node = self._find_node_by_address(trace, sale_contract_addr, buyer_payment_amount)
        if not sale_contract_node:
            self.logger.debug(f"   ❌ Sale contract node not found: {sale_contract_addr[-20:]}")
            return flows, None
        
        # LT window for temporal restriction (override if provided for sale-only fallback)
        if override_window:
            start_lt, end_lt = override_window
            self.logger.debug(f"   ⏰ LT window (override): {start_lt} → {end_lt}")
            
            # CRITICAL: Even with override, ensure sale contract LT is within window
            sale_lt = sale_contract_node.get('transaction', {}).get('lt', 0)
            if sale_lt:
                if sale_lt < start_lt:
                    self.logger.debug(f"   🔧 Override adjusted start_lt for sale LT: {start_lt} -> {sale_lt-3}")
                    start_lt = sale_lt - 3
                elif sale_lt > end_lt:
                    self.logger.debug(f"   🔧 Override adjusted end_lt for sale LT: {end_lt} -> {sale_lt+3}")
                    end_lt = sale_lt + 3
        else:
            start_lt = sale_contract_node.get('transaction', {}).get('lt', 0)
            end_lt = max((seed.get('transaction', {}).get('lt', 0) for seed in nft_seeds), default=start_lt)
            self.logger.debug(f"   ⏰ LT window: {start_lt} → {end_lt}")
        
        # CRITICAL: Ensure sale contract LT is always within window
        sale_lt = sale_contract_node.get('transaction', {}).get('lt', 0)
        if sale_lt:
            if sale_lt < start_lt:
                self.logger.debug(f"   🔧 Adjusting start_lt to include sale LT: {start_lt} -> {sale_lt-3}")
                start_lt = sale_lt - 3
            elif sale_lt > end_lt:
                self.logger.debug(f"   🔧 Adjusting end_lt to include sale LT: {end_lt} -> {sale_lt+3}")
                end_lt = sale_lt + 3
        
        # FRAGMENT LT EXTENSION: After simple_send NFT payment detection, extend window significantly
        if getattr(self, '_fragment_nft_payment_detected', False):
            nft_lt = max((seed.get('transaction', {}).get('lt', 0) for seed in nft_seeds), default=0)
            if nft_lt > 0:
                # Extend window by 5M for Fragment delayed settlements (fees can be 4M+ LT units later)
                extended_end_lt = max(nft_lt, sale_lt, end_lt) + 5000000
                self.logger.debug(f"   🕒 Fragment NFT payment detected: extending LT window {end_lt} → {extended_end_lt} (+5M)")
                end_lt = extended_end_lt
        
        # C) Collect descendants via both routing methods
        descendants_via_source = self._collect_descendants_within_window(
            sale_contract_node, by_source_index, start_lt, end_lt
        )
        descendants_via_dest = self._collect_descendants_via_outmsgs_within_window(
            sale_contract_node, by_dest_index, start_lt, end_lt
        )
        self.logger.debug(f"   🔎 Sale-descendants: source={len(descendants_via_source)}, dest={len(descendants_via_dest)}")
        
        # C.5) FRAGMENT SUPPORT: Add descendants of NFT seeds for Fragment auction flows
        # In Fragment auctions: Buyer → NFT → [Sale Contract, Fragment Fee, Seller]
        # Fragment settlements can be delayed, so use extended window for NFT descendants
        extended_end_lt = end_lt + 50000  # Extended window for Fragment delayed settlements
        descendants_of_seeds = set()
        for seed in nft_seeds:
            seed_descendants_src = self._collect_descendants_within_window(seed, by_source_index, start_lt, extended_end_lt)
            seed_descendants_dest = self._collect_descendants_via_outmsgs_within_window(seed, by_dest_index, start_lt, extended_end_lt)
            descendants_of_seeds.update(seed_descendants_src | seed_descendants_dest)
        self.logger.debug(f"   🌱 Seed-descendants: {len(descendants_of_seeds)} (extended window: {end_lt}→{extended_end_lt})")
        
        # D) Allowed nodes = Union of all paths (comprehensive coverage including NFT descendants)
        descendants_of_sale = descendants_via_source | descendants_via_dest
        allowed_nodes = ancestors_of_seeds | descendants_of_sale | descendants_of_seeds
        
        # FRAGMENT LT EXTENSION: If NFT descendants found, extend traversal window for Fragment settlements
        if descendants_of_seeds and len(descendants_of_seeds) > len(descendants_of_sale):
            end_lt = extended_end_lt  # Use extended window for traversal too
            self.logger.debug(f"   🕒 Extended traversal window to {end_lt} for Fragment NFT descendants")
        
        self.logger.debug(f"   ✅ Allowed union: {len(allowed_nodes)} (ancestors={len(ancestors_of_seeds)}, sale_desc={len(descendants_of_sale)}, seed_desc={len(descendants_of_seeds)})")
        
        # FALLBACK: If no descendants, expand window and retry
        if not descendants_of_sale:
            expanded_start_lt, expanded_end_lt = self._build_expanded_lt_window(
                trace, sale_contract_addr, start_lt, end_lt, by_source_index, by_dest_index
            )
            if expanded_start_lt < start_lt or expanded_end_lt > end_lt:
                self.logger.debug(f"   🛟 EXPAND: descendants=0 → retry with [{expanded_start_lt}, {expanded_end_lt}]")
                start_lt, end_lt = expanded_start_lt, expanded_end_lt
                descendants_via_source = self._collect_descendants_within_window(
                    sale_contract_node, by_source_index, start_lt, end_lt
                )
                descendants_via_dest = self._collect_descendants_via_outmsgs_within_window(
                    sale_contract_node, by_dest_index, start_lt, end_lt
                )
                descendants_of_sale = descendants_via_source | descendants_via_dest
                
                # Re-compute NFT descendants with expanded window (use extended window for Fragment)
                extended_end_lt_fallback = end_lt + 50000
                descendants_of_seeds = set()
                for seed in nft_seeds:
                    seed_descendants_src = self._collect_descendants_within_window(seed, by_source_index, start_lt, extended_end_lt_fallback)
                    seed_descendants_dest = self._collect_descendants_via_outmsgs_within_window(seed, by_dest_index, start_lt, extended_end_lt_fallback)
                    descendants_of_seeds.update(seed_descendants_src | seed_descendants_dest)
                
                allowed_nodes = ancestors_of_seeds | descendants_of_sale | descendants_of_seeds
                self.logger.debug(f"   🔄 Post-expand descendants: sale={len(descendants_of_sale)}, seed={len(descendants_of_seeds)} (total allowed: {len(allowed_nodes)})")
        
        # Build path_nodes for marketplace utils (address, lt pairs for node relocation)
        def _extract_path_nodes(trace, allowed_node_ids, start_lt, end_lt):
            path_nodes = []
            
            def visit_node(node):
                if not isinstance(node, dict):
                    return
                if id(node) not in allowed_node_ids:
                    return
                    
                tx = node.get('transaction', {})
                if not tx:
                    return
                    
                lt = tx.get('lt', 0)
                if lt < start_lt or lt > end_lt:
                    return
                    
                addr = tx.get('account', {}).get('address', '')
                if addr:
                    path_nodes.append({
                        "address": addr,
                        "lt": lt
                    })
            
            UnifiedTraversal.simple_traverse(trace, visit_node)
            return path_nodes
        
        path_nodes = _extract_path_nodes(trace, allowed_nodes, start_lt, end_lt)
        
        # Get event_id from either current context or trace
        event_id = getattr(self, '_current_event_id', trace.get('transaction', {}).get('hash', 'unknown'))
        
        # Set analysis scope for fee collection with extended data for marketplace utils
        self._current_analysis_scope = AnalysisScope(
            allowed_node_ids=allowed_nodes,
            start_lt=start_lt,
            end_lt=end_lt,
            sale_contract_addr=sale_contract_addr,
            nft_address=nft_address,
            # Extended fields for marketplace utils
            anchors={
                "sale_contract": sale_contract_addr,
                "nft_item": nft_address,
                "buyer_wallet": "",  # Will be filled later when determined
                "seller_wallet": ""  # Will be filled later when determined
            },
            path_nodes=path_nodes,
            event_id=event_id,
            scope_strategy="single"
        )
        
        self.logger.debug(f"   ⬇️  Starting downward trace from sale contract: {sale_contract_addr[-20:]}")
        
        # Originalfenster merken
        orig_start_lt, orig_end_lt = start_lt, end_lt
        
        # PRE-PASS SELLER-RESCUE: If no seller credits yet, try to find one candidate before traverse (skip if unlimited)
        if not unlimited_budget and (remaining_budget_check := {'amount': buyer_payment_amount}):  # Copy for check
            if remaining_budget_check['amount'] > max(1e8, 0.01 * buyer_payment_amount):
                # Build expanded LT window dynamically based on graph neighbors
                expanded_start_lt, expanded_end_lt = self._build_expanded_lt_window(
                    trace, sale_contract_addr, start_lt, end_lt, by_source_index, by_dest_index
                )
                
                rescue_id = self._allow_top_credit_candidate_strict(
                    trace, expanded_start_lt, expanded_end_lt, remaining_budget_check, 
                    allowed_nodes, sale_contract_addr, nft_address
                )
                if rescue_id:
                    allowed_nodes.add(rescue_id)
                    
                    # Fenster umstellen für alle nachfolgenden Operationen
                    start_lt, end_lt = expanded_start_lt, expanded_end_lt
                    self.logger.debug(f"   🛟 PRE-PASS: Added seller-rescue candidate and extended traversal window "
                                     f"({orig_start_lt}->{start_lt}, {orig_end_lt}->{end_lt})")
                    
                    # Descendants neu berechnen mit erweitertem Fenster
                    descendants_of_sale = self._collect_descendants_within_window(
                        sale_contract_node, by_source_index, start_lt, end_lt
                    )
                    
                    # Re-compute NFT descendants with expanded window (use extended window for Fragment)
                    extended_end_lt_prepass = end_lt + 50000
                    descendants_of_seeds = set()
                    for seed in nft_seeds:
                        seed_descendants_src = self._collect_descendants_within_window(seed, by_source_index, start_lt, extended_end_lt_prepass)
                        seed_descendants_dest = self._collect_descendants_via_outmsgs_within_window(seed, by_dest_index, start_lt, extended_end_lt_prepass)
                        descendants_of_seeds.update(seed_descendants_src | seed_descendants_dest)
                    
                    allowed_nodes = ancestors_of_seeds | descendants_of_sale | descendants_of_seeds | {rescue_id}
                    self.logger.debug(f"   🛟 PRE-PASS: Recalculated allowed nodes with rescue inclusion")
        
        # Logging für verwendetes Fenster
        self.logger.debug(f"   ⏰ LT window (used): {start_lt} → {end_lt} (orig: {orig_start_lt}→{orig_end_lt})")
        
        # E) Start flow-based traversal with restrictions and NFT path prioritization
        # Use unlimited budget for sale-only fallback scenarios
        if unlimited_budget:
            remaining_budget = {'amount': 10**18}  # Effectively unlimited for sale-only traces
            self.logger.debug(f"   💰 Using unlimited budget for sale-only traversal")
        else:
            remaining_budget = {'amount': buyer_payment_amount}  # Mutable budget for budget-DFS
        self._traverse_outgoing_flows_restricted(sale_contract_node, flows, visited_nodes, nft_address, 
                                                by_source_index, allowed_nodes, start_lt, end_lt, 
                                                remaining_budget, sale_contract_addr, ancestors_of_seeds, 
                                                unlimited_budget)
        
        # F) Safety net: if significant budget remains, try to find large seller credit (skip if unlimited)
        if not unlimited_budget and remaining_budget['amount'] > max(1e8, 0.01 * buyer_payment_amount):
            self.logger.debug(f"   💰 Significant budget remains: {remaining_budget['amount'] / 1e9:.6f} TON")
            top_credit_candidate = self._allow_top_credit_candidate(trace, start_lt, end_lt, remaining_budget, allowed_nodes)
            if top_credit_candidate:
                self.logger.debug(f"   🎯 Adding top credit candidate to allowed nodes")
                allowed_nodes.add(top_credit_candidate)
                # One more traversal round from sale contract with expanded allowed set
                self._traverse_outgoing_flows_restricted(sale_contract_node, flows, visited_nodes, nft_address, 
                                                        by_source_index, allowed_nodes, start_lt, end_lt, 
                                                        remaining_budget, sale_contract_addr, ancestors_of_seeds, 
                                                        unlimited_budget)
        
        self.logger.debug(f"   ⬇️  Downward trace collected {len(flows)} flows")
        
        # DESCENDANT WINDOW EXPANSION: If no flows found, try broader LT window once
        if len(flows) == 0 and not unlimited_budget:
            self.logger.debug(f"   🔄 No descendant flows found, expanding LT window for retry")
            
            # Expand end_lt significantly for post-settlement flows
            expanded_end_lt = end_lt + 50000  # Much broader buffer for post-settlement
            self.logger.debug(f"   ⏰ Expanding LT window for descendants: {start_lt} → {expanded_end_lt} (was {end_lt})")
            
            # Clear visited nodes for fresh traversal
            visited_nodes.clear()
            
            # Retry traversal with expanded window
            self._traverse_outgoing_flows_restricted(sale_contract_node, flows, visited_nodes, nft_address, 
                                                    by_source_index, allowed_nodes, start_lt, expanded_end_lt, 
                                                    remaining_budget, sale_contract_addr, ancestors_of_seeds, 
                                                    unlimited_budget)
            
            self.logger.debug(f"   🔄 Post-expansion trace collected {len(flows)} flows")
            
            # Update end_lt for subsequent operations
            end_lt = expanded_end_lt
            
            # ALIGN FEE WINDOW: Update analysis scope to match expanded flow window
            if hasattr(self, '_current_analysis_scope') and self._current_analysis_scope:
                self._current_analysis_scope.end_lt = expanded_end_lt
                self.logger.debug(f"   🔧 Aligned fee window with expanded flow window: {expanded_end_lt}")
        
        # Store variables for potential post-pass (use current window, possibly extended)
        self._last_remaining_budget = remaining_budget
        self._last_start_lt = start_lt  # May be extended if rescue was used
        self._last_end_lt = end_lt      # May be extended if rescue was used
        self._last_buyer_payment = buyer_payment_amount
        self._last_sale_contract_node = sale_contract_node
        self._last_by_source_index = by_source_index
        self._last_nft_path_hint = ancestors_of_seeds
        self._last_allowed_nodes = allowed_nodes.copy()  # Copy the set
        
        # Return current analysis scope
        current_scope = getattr(self, '_current_analysis_scope', None)
        return flows, current_scope
    
    def _precompute_allowed_nodes(self, trace: Dict, nft_seeds: List[Dict]) -> set:
        """Precompute allowed nodes: only nodes on paths leading to NFT seeds"""
        from collections import defaultdict
        allowed = set()
        
        # Build reverse index: account_addr -> [nodes with that address]
        by_account = defaultdict(list)
        def index_by_account(node):
            if isinstance(node, dict):
                tx = node.get('transaction', {})
                if tx:
                    account_addr = self._normalize_address(tx.get('account', {}).get('address', ''))
                    if account_addr:
                        by_account[account_addr].append(node)
        
        UnifiedTraversal.simple_traverse(trace, index_by_account)
        
        # Backwards traversal: from each seed, mark path back to source
        def mark_path_backwards(node):
            if not isinstance(node, dict) or id(node) in allowed:
                return
                
            allowed.add(id(node))  # Use object id for exact matching
            
            tx = node.get('transaction', {})
            if not tx:
                return
                
            in_msg = tx.get('in_msg', {})
            if not in_msg:
                return
                
            source = in_msg.get('source', {})
            if not source:
                return
                
            source_addr = self._normalize_address(source.get('address', ''))
            if not source_addr:
                return
            
            # Find predecessor node(s) with this source address
            current_lt = tx.get('lt', 0)
            candidates = by_account.get(source_addr, [])
            
            # Take the node with highest LT that's still <= current_lt (true predecessor)
            valid_predecessors = [n for n in candidates 
                                if n.get('transaction', {}).get('lt', 0) <= current_lt]
            
            if valid_predecessors:
                predecessor = max(valid_predecessors, 
                                key=lambda n: n.get('transaction', {}).get('lt', 0))
                mark_path_backwards(predecessor)
        
        # Mark paths from all NFT seeds
        for seed in nft_seeds:
            mark_path_backwards(seed)
        
        return allowed
    
    def _collect_descendants_within_window(self, root_node: Dict, by_source_index: Dict, 
                                         start_lt: int, end_lt: int) -> set:
        """Collect all descendants of root node within LT window using BFS"""
        allowed = set()
        stack = [root_node]
        
        while stack:
            node = stack.pop()
            tx = node.get('transaction', {})
            if not tx:
                continue
                
            lt = tx.get('lt', 0)
            if not (start_lt <= lt <= end_lt):
                continue
                
            node_id = id(node)
            if node_id in allowed:
                continue
                
            allowed.add(node_id)
            
            # Find children via by_source_index
            account_addr = self._normalize_address(tx.get('account', {}).get('address', ''))
            children = by_source_index.get(account_addr, [])
            
            for child in children:
                stack.append(child)
        
        return allowed
    
    def _collect_descendants_via_outmsgs_within_window(self, root_node: Dict, by_dest_index: Dict, 
                                                      start_lt: int, end_lt: int) -> set:
        """Collect descendants via out_msgs.destination routing within LT window"""
        allowed = set()
        stack = [root_node]
        
        while stack:
            node = stack.pop()
            tx = node.get('transaction', {})
            if not tx:
                continue
                
            lt = tx.get('lt', 0)
            if not (start_lt <= lt <= end_lt):
                continue
                
            node_id = id(node)
            if node_id in allowed:
                continue
                
            allowed.add(node_id)
            
            # Find children via by_dest_index (out_msgs.destination routing)
            account_addr = self._normalize_address(tx.get('account', {}).get('address', ''))
            children = by_dest_index.get(account_addr, [])
            
            for child in children:
                stack.append(child)
        
        return allowed
    
    def _traverse_outgoing_flows_restricted(self, node: Dict, flows: List[Dict], visited: set, 
                                          nft_address: str, by_source_index: Dict, allowed_nodes: set,
                                          start_lt: int, end_lt: int, remaining_budget: Dict, 
                                          sale_contract_addr: str, nft_path_hint: set = None, 
                                          unlimited_budget: bool = False):
        """Follow outgoing flows with NFT-specific restrictions and budget control"""
        if not isinstance(remaining_budget, dict):
            remaining_budget = {'amount': remaining_budget}  # Make it mutable
            
        tx = node.get('transaction', {})
        if not tx:
            return
            
        # Check if node is on allowed path (intersection constraint)
        if id(node) not in allowed_nodes:
            self.logger.debug(f"   ⛔ Node not on NFT path: {tx.get('account', {}).get('address', 'unknown')[-20:]}")
            return
            
        account_addr = tx.get('account', {}).get('address', '')
        tx_lt = tx.get('lt', 0)
        node_key = f"{account_addr}:{tx_lt}"
        
        # Check LT window (temporal constraint)
        if not (start_lt <= tx_lt <= end_lt):
            self.logger.debug(f"   ⏰ Node outside LT window: {tx_lt} not in [{start_lt}, {end_lt}]")
            return
        
        # Check budget (prevent over-collection)
        epsilon = max(1e8, 0.005 * abs(remaining_budget['amount']))  # 0.1 TON or 0.5% tolerance
        if remaining_budget['amount'] <= epsilon:
            self.logger.debug(f"   💰 Budget exhausted: {remaining_budget['amount'] / 1e9:.6f} TON remaining")
            return
            
        self.logger.debug(f"   🔍 Traversing node: {account_addr[-20:] if account_addr else 'unknown'} (LT: {tx_lt}, Budget: {remaining_budget['amount'] / 1e9:.3f})")
        
        if node_key in visited:
            self.logger.debug(f"   ⏭️  Already visited: {account_addr[-20:]}:{tx_lt}")
            return
        visited.add(node_key)
        
        # Collect credit_phase from this node
        if 'credit_phase' in tx and 'credit' in tx['credit_phase']:
            try:
                credit_nanoton = int(tx['credit_phase']['credit'])
                self.logger.debug(f"   💰 Credit phase: {credit_nanoton / 1e9:.6f} TON")
                if credit_nanoton > 0:  # Only positive credits
                    # Set current context for enhanced buyer change detection
                    self._current_flow_credit = credit_nanoton
                    # Get marketplace fee sum from flows collected so far
                    self._current_marketplace_fee_sum = sum(f.get('amount_nanoton', 0) / 1e9 for f in flows if f.get('flow_type') == 'marketplace_fee')
                    
                    # Classify flow type (with sale_contract distinction)
                    flow_type = self._classify_flow_type_enhanced(account_addr, nft_address, sale_contract_addr)
                    
                    # Log buyer change detection
                    if flow_type == 'buyer_change':
                        self.logger.debug(f"   🔁 CHANGE filtered (→ buyer): +{credit_nanoton/1e9:.6f} TON")
                    
                    if flow_type == 'sale_contract_credit':
                        # 👉 Anker ja, aber KEIN Flow-Record, sonst doppelte Positivseite
                        self.logger.debug(
                            f"   🧲 SOURCE CREDIT (ignored for balance): "
                            f"{account_addr[-20:]} = {credit_nanoton / 1e9:.6f} TON"
                        )
                        # WICHTIG: kein Budget-Abzug hier - das ist die Quelle
                    else:
                        # Check for fee deduplication if this is a marketplace fee
                        lt = node.get('transaction', {}).get('lt', 0)
                        
                        if flow_type == 'marketplace_fee':
                            # STRENGTHENED deduplication: check before adding
                            if not hasattr(self, '_seen_fee_keys'):
                                self._seen_fee_keys = set()
                            dedupe_key = (account_addr, lt, credit_nanoton)
                            
                            # Skip if already seen
                            if dedupe_key in self._seen_fee_keys:
                                self.logger.debug(f"   ⏭️  SKIP duplicate fee in flow traversal: {dedupe_key}")
                                return  # Skip this entire credit phase processing
                                
                            self._seen_fee_keys.add(dedupe_key)
                        
                        # NEW: Budget-Clamping - hart begrenzen auf verfügbares Budget
                        # Dedupe protection
                        if not hasattr(self, '_credited_keys'):
                            self._credited_keys = set()
                        dedupe_key = (self._normalize_address(account_addr), tx_lt)
                        
                        if dedupe_key in self._credited_keys:
                            raw = 0  # Schon verbucht
                        else:
                            self._credited_keys.add(dedupe_key)
                            raw = int(credit_nanoton)
                        
                        # Budget-Clamping anwenden (nur bei echten Credits, nicht NFT-interne Hops)
                        if flow_type == 'nft_contract':
                            take = raw  # NFT credits don't consume budget
                        else:
                            take = _take_budget(remaining_budget, raw)
                        
                        if take > 0:
                            flows.append({
                                'account': {'address': account_addr},
                                'amount_nanoton': take,
                                'flow_type': flow_type,
                                'evidence': f'clamped from {raw/1e9:.6f} TON'
                            })
                            self.logger.debug(
                                f"   💳 FLOW CREDIT: {account_addr[-20:]} = "
                                f"{take / 1e9:.6f} TON ({flow_type})"
                            )
                            
                            # Add mirroring debit for sale contract to maintain balance (sale-only scenarios)
                            if unlimited_budget and sale_contract_addr:
                                flows.append({
                                    'account': {'address': sale_contract_addr},
                                    'amount_nanoton': -take,
                                    'flow_type': 'sale_contract_debit',
                                    'evidence': f'Mirroring debit for sale contract: -{take / 1e9:.6f} TON'
                                })
                                self.logger.debug(f"   🔄 MIRRORING DEBIT: {sale_contract_addr[-20:]} = -{take / 1e9:.6f} TON")
                        
                        # WICHTIG: Sobald Budget erschöpft, keine weiteren Kinder verfolgen  
                        if remaining_budget.get("amount", 0) <= 0:
                            self.logger.debug(f"   🛑 Budget exhausted, stopping traversal")
                            return
                        
                        # 🌟 Clamp bei mini Overshoot (Rundungstoleranz)
                        buyer_payment_basis = abs(getattr(self, '_last_buyer_payment', 0)) or credit_nanoton or 1e9
                        epsilon = max(1e8, 0.005 * buyer_payment_basis)
                        if remaining_budget['amount'] < 0 and abs(remaining_budget['amount']) <= epsilon:
                            self.logger.debug(
                                f"   🔧 Clamp tiny overshoot: {remaining_budget['amount']/1e9:.6f} TON → 0"
                            )
                            remaining_budget['amount'] = 0
                        
                        self.logger.debug(f"   💰 Budget after deduction: {remaining_budget['amount'] / 1e9:.6f} TON")
                        
            except (ValueError, TypeError) as e:
                self.logger.debug(f"   ❌ Credit parse error: {e}")
        else:
            self.logger.debug(f"   ⚠️  No credit_phase in transaction")
        
        # Continue to children if budget allows, with NFT path prioritization
        account_normalized = self._normalize_address(account_addr)
        
        # Unite children from both indices (source and destination routing)
        children_src = by_source_index.get(account_normalized, [])
        children_dest = getattr(self, '_by_dest_index', {}).get(account_normalized, [])
        
        # FRAGMENT FIX: Also check for direct children in the trace node (Fragment sales pattern)
        direct_children = []
        if 'children' in node:
            direct_children = node['children']
            self.logger.debug(f"   🔗 FRAGMENT: Found {len(direct_children)} direct children in node {account_addr[-20:]}")
        
        # Deduplicate by node id to avoid processing same node twice
        all_children_sources = children_src + children_dest + direct_children
        children_all = list({id(n): n for n in all_children_sources}.values())
        self.logger.debug(f"   🔗 Found {len(children_all)} child flows from {account_addr[-20:]} (src: {len(children_src)}, dest: {len(children_dest)}, direct: {len(direct_children)})")
        
        # Prioritize children on NFT path (from nft_path_hint) first
        def on_nft_path(node):
            return nft_path_hint and id(node) in nft_path_hint
        
        # Sort: NFT path children first, then by LT
        children_sorted = sorted(children_all, key=lambda n: (not on_nft_path(n), 
                                                             n.get('transaction', {}).get('lt', 0)))
        
        for child_node in children_sorted:
            child_tx = child_node.get('transaction', {})
            if child_tx:
                child_addr = child_tx.get('account', {}).get('address', '')
                child_lt = child_tx.get('lt', 0)
                child_key = f"{child_addr}:{child_lt}"
                if child_key not in visited:
                    self._traverse_outgoing_flows_restricted(child_node, flows, visited, nft_address, 
                                                           by_source_index, allowed_nodes, start_lt, end_lt, 
                                                           remaining_budget, sale_contract_addr, nft_path_hint, 
                                                           unlimited_budget)
    
    def _classify_flow_type_enhanced(self, address: str, nft_address: str, sale_contract_addr: str) -> str:
        """Enhanced flow classification with sale_contract distinction and buyer change detection"""
        addr_normalized = self._normalize_address(address)
        
        # Check if it's the sale contract itself
        if self._normalize_address(sale_contract_addr) == addr_normalized:
            return 'sale_contract_credit'
        
        # Check marketplace addresses from config (highest priority)
        if addr_normalized in {self._normalize_address(addr) for addr in self.marketplace_addresses}:
            return 'marketplace_fee'
        
        # Check if it's the NFT contract itself
        if self._normalize_address(nft_address) == addr_normalized:
            return 'nft_contract'
        
        # Check if this is buyer change (Rückgeld/Change zum Buyer)
        buyer_addr_norm = self._normalize_address(getattr(self, '_detected_buyer_wallet', ''))
        if buyer_addr_norm and addr_normalized == buyer_addr_norm:
            # ENHANCED BUYER CHANGE DETECTION: Check if this looks like seller payout instead
            # If amount ≈ (buyer_payment - marketplace_fees), it's likely seller payout, not buyer change
            current_credit = getattr(self, '_current_flow_credit', 0) / 1e9  # Convert to TON
            buyer_payment = getattr(self, '_last_buyer_payment', 0) / 1e9 if hasattr(self, '_last_buyer_payment') else 0
            marketplace_fees = getattr(self, '_current_marketplace_fee_sum', 0)
            
            # Include Fragment fees from relaxed scope (fee_hits)
            if hasattr(self, 'fee_hits') and marketplace_fees == 0:
                fragment_fees = sum(h.amount_nanoton for h in self.fee_hits 
                                  if h.marketplace_type == 'fragment')
                if fragment_fees > 0:
                    marketplace_fees = fragment_fees / 1e9
            
            if buyer_payment > 0 and marketplace_fees > 0:
                expected_seller_amount = buyer_payment - marketplace_fees
                tolerance = max(0.05, expected_seller_amount * 0.02)  # 2% or 0.05 TON tolerance
                
                if abs(current_credit - expected_seller_amount) <= tolerance:
                    self.logger.debug(f"   🎯 BUYER CHANGE OVERRIDE: {current_credit:.6f} TON ≈ expected seller ({expected_seller_amount:.6f} TON, diff: {abs(current_credit - expected_seller_amount):.6f}) → seller_credit")
                    return 'seller_credit'
            
            return 'buyer_change'
        
        # Default to seller credit for wallets
        return 'seller_credit'
    
    def _allow_top_credit_candidate(self, trace: Dict, start_lt: int, end_lt: int, 
                                   remaining_budget: Dict, exclude_ids: set) -> Optional[int]:
        """Safety net: find the best credit candidate when large budget remains"""
        target = abs(remaining_budget['amount'])
        best = (None, float('inf'))
        
        def scan_node(node):
            nonlocal best
            if not isinstance(node, dict):
                return
            
            tx = node.get('transaction', {})
            if not tx:
                return
                
            # Check LT window
            lt = tx.get('lt', 0)
            if not (start_lt <= lt <= end_lt):
                return
                
            # Skip already allowed nodes
            node_id = id(node)
            if node_id in exclude_ids:
                return
                
            # Check for positive credit
            credit_phase = tx.get('credit_phase', {})
            if 'credit' not in credit_phase:
                return
                
            try:
                credit_amount = int(credit_phase['credit'])
            except (ValueError, TypeError):
                return
                
            if credit_amount <= 0:
                return
                
            # Calculate difference from remaining budget
            diff = abs(credit_amount - target)
            tolerance = max(1e9, 0.02 * max(target, 1))  # max(1 TON, 2% of target) - more generous
            
            if diff <= tolerance and diff < best[1]:
                account_addr = tx.get('account', {}).get('address', '')
                self.logger.debug(f"   🎯 Credit candidate: {account_addr[-20:]} credit={credit_amount/1e9:.6f} TON (diff: {diff/1e9:.6f})")
                best = (node_id, diff)
        
        # Scan entire trace for credit candidates
        UnifiedTraversal.simple_traverse(trace, scan_node)
        
        # If no candidate found within tolerance, try fallback: largest positive credit
        if best[0] is None:
            best_fallback = (None, 0)
            
            def scan_fallback(node):
                nonlocal best_fallback
                if not isinstance(node, dict):
                    return
                tx = node.get('transaction', {})
                if not tx:
                    return
                lt = tx.get('lt', 0)
                if not (start_lt <= lt <= end_lt):
                    return
                node_id = id(node)
                if node_id in exclude_ids:
                    return
                
                credit_phase = tx.get('credit_phase', {})
                if 'credit' not in credit_phase:
                    return
                try:
                    credit_amount = int(credit_phase['credit'])
                except (ValueError, TypeError):
                    return
                if credit_amount <= 0:
                    return
                
                # Exclude fee addresses and NFT contract
                addr = self._normalize_address(tx.get('account', {}).get('address', ''))
                if addr in {self._normalize_address(a) for a in self.marketplace_addresses}:
                    return
                if addr == self._normalize_address(getattr(self, '_current_analysis_scope', AnalysisScope(set(),0,0,'','')).nft_address):
                    return
                    
                if credit_amount > best_fallback[1]:
                    best_fallback = (node_id, credit_amount)
            
            UnifiedTraversal.simple_traverse(trace, scan_fallback)
            
            if best_fallback[0] is not None:
                self.logger.debug(f"   🛟 Fallback: largest positive credit as seller-rescue candidate ({best_fallback[1]/1e9:.6f} TON)")
                return best_fallback[0]
        
        if best[0] is not None:
            self.logger.debug(f"   ✅ Selected top credit candidate with difference: {best[1]/1e9:.6f} TON")
        
        return best[0]
    
    def _build_by_source_index(self, trace: Dict) -> Dict[str, List[Dict]]:
        """Build index mapping source addresses to child nodes for O(1) lookup"""
        from collections import defaultdict
        by_source = defaultdict(list)
        
        def index_node(node):
            if not isinstance(node, dict):
                return
            
            tx = node.get('transaction', {})
            if tx:
                in_msg = tx.get('in_msg', {})
                if in_msg:
                    source = in_msg.get('source', {})
                    if source:
                        source_addr = self._normalize_address(source.get('address', ''))
                        if source_addr:
                            by_source[source_addr].append(node)
        
        UnifiedTraversal.simple_traverse(trace, index_node)
        self.logger.debug(f"   📇 Built by_source index with {len(by_source)} unique sources")
        return dict(by_source)
    
    def _build_by_dest_index(self, trace: Dict) -> Dict[str, List[Dict]]:
        """Build index mapping source accounts to child nodes via out_msgs.destination"""
        from collections import defaultdict
        by_dest = defaultdict(list)
        
        # Build helper index: account -> nodes (for destination resolution)
        acc2nodes = defaultdict(list)
        def collect_accounts(node):
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if tx:
                acc = self._normalize_address(tx.get('account', {}).get('address', ''))
                if acc:
                    acc2nodes[acc].append(node)
        
        UnifiedTraversal.simple_traverse(trace, collect_accounts)
        
        # Build destination index
        def index_node(node):
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
            src = self._normalize_address(tx.get('account', {}).get('address', ''))
            if not src:
                return
            
            # Check all out_msgs for destinations
            out_msgs = tx.get('out_msgs', [])
            if not isinstance(out_msgs, list):
                out_msgs = [out_msgs] if out_msgs else []
            
            for msg in out_msgs:
                if not isinstance(msg, dict):
                    continue
                dest_info = msg.get('destination', {})
                if not isinstance(dest_info, dict):
                    continue
                dest = self._normalize_address(dest_info.get('address', ''))
                if not dest:
                    continue
                    
                # Find all nodes that represent this destination account
                for child in acc2nodes.get(dest, []):
                    by_dest[src].append(child)
        
        UnifiedTraversal.simple_traverse(trace, index_node)
        self.logger.debug(f"   📇 Built by_dest index with {len(by_dest)} unique sources")
        return dict(by_dest)
    
    def _traverse_outgoing_flows_indexed(self, node: Dict, flows: List[Dict], visited: set, nft_address: str, by_source_index: Dict):
        """Follow outgoing flows using pre-built index for O(1) child lookup"""
        tx = node.get('transaction', {})
        if not tx:
            self.logger.debug(f"   ⚠️  No transaction in node")
            return
            
        account_addr = tx.get('account', {}).get('address', '')
        tx_lt = tx.get('lt', 0)
        node_key = f"{account_addr}:{tx_lt}"
        self.logger.debug(f"   🔍 Traversing node: {account_addr[-20:] if account_addr else 'unknown'} (LT: {tx_lt})")
        
        if node_key in visited:
            self.logger.debug(f"   ⏭️  Already visited: {account_addr[-20:]}:{tx_lt}")
            return
        visited.add(node_key)
        
        # Collect credit_phase from this node
        if 'credit_phase' in tx and 'credit' in tx['credit_phase']:
            try:
                credit_nanoton = int(tx['credit_phase']['credit'])
                self.logger.debug(f"   💰 Credit phase: {credit_nanoton / 1e9:.6f} TON")
                if credit_nanoton > 0:  # Only positive credits
                    flows.append({
                        'account': {'address': account_addr},
                        'amount_nanoton': credit_nanoton,
                        'flow_type': self._classify_flow_type(account_addr, nft_address),
                        'evidence': f'Flow-based credit: {credit_nanoton / 1e9:.6f} TON'
                    })
                    self.logger.debug(f"   💳 FLOW CREDIT: {account_addr[-20:]} = {credit_nanoton / 1e9:.6f} TON")
            except (ValueError, TypeError) as e:
                self.logger.debug(f"   ❌ Credit parse error: {e}")
        else:
            self.logger.debug(f"   ⚠️  No credit_phase in transaction")
        
        # Use index to find children efficiently (O(1) instead of O(N))
        account_normalized = self._normalize_address(account_addr)
        children = by_source_index.get(account_normalized, [])
        self.logger.debug(f"   🔗 Found {len(children)} child flows from {account_addr[-20:]}")
        
        for child_node in children:
            child_tx = child_node.get('transaction', {})
            if child_tx:
                child_addr = child_tx.get('account', {}).get('address', '')
                child_lt = child_tx.get('lt', 0)
                child_key = f"{child_addr}:{child_lt}"
                if child_key not in visited:
                    self._traverse_outgoing_flows_indexed(child_node, flows, visited, nft_address, by_source_index)
    
    def _find_node_by_address(self, trace: Dict, target_addr: str, expected_credit: int = None) -> Dict:
        """Find the best matching node by account address, optionally by credit amount"""
        target_normalized = self._normalize_address(target_addr)
        candidates = []
        
        def search_node(node):
            if not isinstance(node, dict):
                return
                
            tx = node.get('transaction', {})
            if tx:
                account_addr = tx.get('account', {}).get('address', '')
                if self._normalize_address(account_addr) == target_normalized:
                    candidates.append(node)
        
        UnifiedTraversal.simple_traverse(trace, search_node)
        
        if not candidates:
            return None
        
        # If we have an expected credit amount, find the best match
        if expected_credit is not None and expected_credit > 0:
            best_node = None
            min_diff = float('inf')
            
            for candidate in candidates:
                tx = candidate.get('transaction', {})
                credit_phase = tx.get('credit_phase', {})
                if 'credit' in credit_phase:
                    try:
                        credit_amount = int(credit_phase['credit'])
                        # Find node with credit closest to expected amount (within reasonable range)
                        diff = abs(credit_amount - expected_credit)
                        tolerance = max(1e8, 0.01 * expected_credit)  # 0.1 TON or 1% tolerance
                        
                        if diff <= tolerance and diff < min_diff:
                            min_diff = diff
                            best_node = candidate
                            self.logger.debug(f"   🎯 Found matching sale contract: credit={credit_amount/1e9:.6f} TON (expected: {expected_credit/1e9:.6f})")
                    except (ValueError, TypeError):
                        continue
            
            if best_node:
                return best_node
        
        # Fallback: return first candidate with positive credit, or just first candidate
        for candidate in candidates:
            tx = candidate.get('transaction', {})
            credit_phase = tx.get('credit_phase', {})
            if 'credit' in credit_phase:
                try:
                    credit_amount = int(credit_phase['credit'])
                    if credit_amount > 0:
                        self.logger.debug(f"   📍 Using sale contract with credit: {credit_amount/1e9:.6f} TON")
                        return candidate
                except (ValueError, TypeError):
                    continue
        
        # Last resort: return first candidate
        self.logger.debug(f"   ⚠️  Using first available sale contract candidate")
        return candidates[0] if candidates else None
    
    def _is_marketplace_contract(self, address: str) -> bool:
        """Check if address is a known marketplace sale/escrow contract"""
        addr_normalized = self._normalize_address(address)
        marketplace_normalized = {self._normalize_address(addr) for addr in self.marketplace_addresses}
        return addr_normalized in marketplace_normalized
    
    def _classify_flow_type(self, address: str, nft_address: str) -> str:
        """Classify flow type based on address characteristics with marketplace priority"""
        return UtilityHelpers.classify_flow_type(
            address=address,
            nft_address=nft_address,
            sale_contract_addr="",  # Not available in simple version
            marketplace_addresses=list(self.marketplace_addresses.keys()),
            normalize_fn=self._normalize_address
        )
    
    def _transaction_references_nft(self, tx: Dict, nft_address: str) -> bool:
        """Check if transaction contains any reference to our NFT"""
        # Check all messages
        for msg_key in ['in_msg', 'out_msgs']:
            if msg_key in tx:
                msgs = tx[msg_key] if isinstance(tx[msg_key], list) else [tx[msg_key]]
                for msg in msgs:
                    if self._msg_references_nft(msg, nft_address):
                        return True
        return False
    
    def _record_fee_hit(self, node: Dict, addr: str, marketplace_type: str, credit_nanoton: int, aborted: bool, lt: int, within_scope: bool = True, linkage: str = "unknown"):
        """Record a fee hit (amount-independent) for marketplace classification"""
        from .config import nanoton_to_ton, TON_PRECISION_CALCULATION
        
        # Enhanced fee deduplication: (addr, lt, amount_nanoton) key
        dedupe_key = (addr, lt, credit_nanoton)
        if hasattr(self, '_seen_fee_keys') and dedupe_key in self._seen_fee_keys:
            self.logger.debug(f"   ⏭️  FEE SKIP (duplicate): {marketplace_type} = {credit_nanoton / 1e9:.6f} TON ({addr[-12:]}) lt={lt}")
            return
        
        # Track this fee hit to prevent duplicates
        if hasattr(self, '_seen_fee_keys'):
            self._seen_fee_keys.add(dedupe_key)
        
        amt_ton = nanoton_to_ton(int(credit_nanoton), TON_PRECISION_CALCULATION)
        fee_hit = FeeHit(
            address=addr,
            marketplace_type=marketplace_type,
            amount_nanoton=int(credit_nanoton),
            amount_ton=amt_ton,
            lt=lt,
            aborted=bool(aborted),
            evidence=f"fee address hit (credit={amt_ton:.6f} TON, aborted={aborted}, LT={lt})",
            within_scope=within_scope,
            linkage=linkage
        )
        self.fee_hits.append(fee_hit)
        scope_status = "scoped" if within_scope else "relaxed"
        self.logger.debug(f"   🏪 FEE HIT ({scope_status}): {marketplace_type} = {amt_ton:.6f} TON ({addr[-12:]}) lt={lt} linkage={linkage}")
    
    def _collect_fee_hits_in_window(self, trace: Dict, allowed_nodes: set, start_lt: int, end_lt: int, use_allowed_nodes: bool = False, within_scope: bool = True):
        """Collect fee hits (amount-independent) within LT window - UNIFIED"""
        # Check if NFT-specific strict collection should be used
        if hasattr(self, '_nft_specific_fee_context') and self._nft_specific_fee_context:
            mode = 'nft_specific'
        else:
            mode = 'normal'
            
        config = FeeCollectionConfig(
            mode=mode,
            start_lt=start_lt,
            end_lt=end_lt,
            allowed_nodes=allowed_nodes,
            use_allowed_nodes=use_allowed_nodes,
            within_scope=within_scope
        )
        fee_hits = self.fee_collector.collect_fees(trace, config)
        self.fee_hits.extend(fee_hits)
    
    def _collect_fee_hits_original(self, trace: Dict, allowed_nodes: set, start_lt: int, end_lt: int, use_allowed_nodes: bool = False, within_scope: bool = True):
        """Original fee collection logic (account-based)"""
        # Normalize marketplace addresses for faster lookup
        fee_whitelist = {self._normalize_address(a): t for a, t in self.marketplace_addresses.items()}
        
        self.logger.debug(f"   🏪 FEE scan start: LT [{start_lt}, {end_lt}], restrict_nodes={use_allowed_nodes}")
        
        def visit_node(node):
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
                
            # Check LT window (always required)
            lt = tx.get('lt', 0)
            if lt < start_lt or lt > end_lt:
                return
                
            # Optional: Check if node is in allowed set
            if use_allowed_nodes and allowed_nodes and id(node) not in allowed_nodes:
                return
                
            # Check if this account is a marketplace fee account
            acc_addr = self._normalize_address(tx.get('account', {}).get('address', ''))
            if acc_addr not in fee_whitelist:
                return
                
            # Record the fee hit regardless of amount and regardless of in_msg validity
            credit = int(tx.get('credit_phase', {}).get('credit', 0) or 0)
            aborted = bool(tx.get('aborted', False))
            marketplace_type = fee_whitelist[acc_addr]
            
            self._record_fee_hit(node, acc_addr, marketplace_type, credit, aborted, lt, within_scope, "account_node")
        
        # Traverse the entire trace to find fee hits (account-based)
        UnifiedTraversal.simple_traverse(trace, visit_node)
    
    def _collect_fee_hits_nft_specific(self, trace: Dict, allowed_nodes: set, start_lt: int, end_lt: int, use_allowed_nodes: bool = False, within_scope: bool = True):
        """NFT-specific fee collection with strict source binding"""
        fee_whitelist = {self._normalize_address(a): t for a, t in self.marketplace_addresses.items()}
        context = self._nft_specific_fee_context
        
        # Calculate tight NFT-specific window
        lt_sale = context['sale_lt']
        lt_nft = context['nft_credit_lt'] 
        lt_seller = context.get('seller_payout_lt', None)
        
        # Tight window: sale -> min(nft_credit, seller_payout) + buffer
        # Fragment sales: larger buffer for child fee transactions
        # Check if this appears to be a Fragment sale (fee address or context hints)
        is_fragment_sale = (
            any(hit.marketplace_type == 'fragment' for hit in getattr(self, 'fee_hits', [])) or
            self._normalize_address(context.get('sale_contract_addr', '')) in 
            {self._normalize_address(a) for a in self.marketplace_addresses if self.marketplace_addresses[a] == 'fragment'}
        )
        buffer_size = 10 if is_fragment_sale else 2  # Larger buffer for Fragment sales
        
        win_start = lt_sale
        if lt_seller and lt_seller < lt_nft:
            win_end = lt_seller + buffer_size  # Buffer after seller payout
        else:
            win_end = lt_nft + buffer_size     # Buffer after NFT credit
        
        # Relaxed window for fallback - extended for Fragment post-settlement fees  
        win_end_relaxed = lt_sale + 5000000  # Much larger buffer for Fragment fees (up to 5 seconds later)
        
        self.logger.debug(f"   🏪 NFT-SPECIFIC FEE scan: tight=[{win_start}, {win_end}], relaxed=[{win_start}, {win_end_relaxed}], sale_contract={context['sale_contract_addr'][-12:]}")
        
        hits_strict = []
        hits_relaxed = []
        
        def visit_node(node):
            nonlocal hits_strict, hits_relaxed
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
                
            lt = tx.get('lt', 0)
            acc_addr = self._normalize_address(tx.get('account', {}).get('address', ''))
            
            # Must be fee account
            if acc_addr not in fee_whitelist:
                return
                
            # Must be in extended relaxed window
            if lt < win_start or lt > win_end_relaxed:
                return
                
            credit = int(tx.get('credit_phase', {}).get('credit', 0) or 0)
            aborted = bool(tx.get('aborted', False))
            marketplace_type = fee_whitelist[acc_addr]
            
            # Check for strict source binding (highest priority)
            in_msg = tx.get('in_msg', {})
            if in_msg and 'source' in in_msg:
                source_addr = self._normalize_address(in_msg.get('source', {}).get('address', ''))
                if (source_addr == self._normalize_address(context['sale_contract_addr']) and 
                    win_start <= lt <= win_end):
                    hits_strict.append((node, acc_addr, marketplace_type, credit, aborted, lt, "source=sale_contract"))
                    self.logger.debug(f"   🎯 STRICT FEE HIT: {marketplace_type} = {credit/1e9:.6f} TON ({acc_addr[-12:]}) lt={lt} source=sale_contract")
                    return
            
            # Check if node is descendant of sale contract (medium priority) 
            # Extended check for Fragment: if sale contract has out_msg to this fee address
            is_descendant = self._is_descendant_of_sale_simple(node, context['sale_contract_addr'], start_lt, end_lt)
            is_child_of_sale = self._is_child_of_sale_contract(node, context['sale_contract_addr'], acc_addr)
            
            if (lt <= win_end_relaxed and id(node) in allowed_nodes and (is_descendant or is_child_of_sale)):
                linkage = "descendant_of_sale" if is_descendant else "child_of_sale"
                linkage_detail = f"{linkage}(tx_lt={lt})"
                hits_relaxed.append((node, acc_addr, marketplace_type, credit, aborted, lt, linkage_detail))
                self.logger.debug(f"   🎯 RELAXED FEE HIT: {marketplace_type} = {credit/1e9:.6f} TON ({acc_addr[-12:]}) lt={lt} {linkage_detail}")
                return
                
            # Fallback: account-based (only in tight window AND on allowed path to avoid cross-contamination)
            if win_start <= lt <= win_end and id(node) in allowed_nodes:
                hits_relaxed.append((node, acc_addr, marketplace_type, credit, aborted, lt, "account_node_tight"))
                self.logger.debug(f"   🎯 FALLBACK FEE HIT: {marketplace_type} = {credit/1e9:.6f} TON ({acc_addr[-12:]}) lt={lt} account_node_tight")
        
        UnifiedTraversal.simple_traverse(trace, visit_node)
        
        # Record hits: prefer strict, then relaxed
        selected_hits = hits_strict if hits_strict else hits_relaxed
        
        # Score and sort hits by proximity to NFT events
        def score_hit(hit_data):
            node, addr, mtype, credit, aborted, lt, linkage = hit_data
            score = 0
            score -= abs(lt - lt_nft) * 2      # Closer to NFT credit = better
            if lt_seller:
                score -= abs(lt - lt_seller)   # Closer to seller payout = better
            if "source=sale_contract" in linkage:
                score += 100                   # Strong preference for source binding
            elif "descendant_of_sale" in linkage:
                score += 50                    # Medium preference for graph linkage
            return score
        
        selected_hits.sort(key=score_hit, reverse=True)
        
        # Record the selected hits
        for node, addr, mtype, credit, aborted, lt, linkage in selected_hits:
            # Fragment fees in relaxed window should still be within_scope if properly linked
            fee_within_scope = within_scope
            if not within_scope and mtype == "fragment" and ("descendant_of_sale" in linkage or "child_of_sale" in linkage):
                fee_within_scope = True  # Fragment post-settlement fees are legitimate
                self.logger.debug(f"   🔧 Fragment fee in relaxed window marked as within_scope: {linkage}")
            
            self._record_fee_hit(node, addr, mtype, credit, aborted, lt, fee_within_scope, linkage)
        
        self.logger.debug(f"   📊 NFT-SPECIFIC: {len(selected_hits)} hits selected (strict={len(hits_strict)}, relaxed={len(hits_relaxed) - len(hits_strict)})")
        
        # Additional scan: message-based fee detection (out_msgs destinations)
        initial_hits = len(self.fee_hits)
        self._scan_messages_for_fee_destinations(trace, fee_whitelist, start_lt, end_lt, within_scope)
        
        self.logger.debug(f"   📊 Fee evidence: {len(self.fee_hits)} hits collected (account: {initial_hits}, messages: {len(self.fee_hits) - initial_hits})")
    
    def _scan_messages_for_fee_destinations(self, trace: Dict, fee_whitelist: Dict[str, str], start_lt: int, end_lt: int, within_scope: bool = True):
        """Scan out_msgs destinations for fee addresses that might not have own nodes"""
        # Track already found addresses to avoid duplicates
        found_addresses = {hit.address for hit in self.fee_hits}
        
        def scan_node_messages(node):
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
                
            # Check LT window
            lt = tx.get('lt', 0)
            if lt < start_lt or lt > end_lt:
                return
                
            # Scan out_msgs for fee destinations
            out_msgs = tx.get('out_msgs', [])
            if not isinstance(out_msgs, list):
                out_msgs = [out_msgs] if out_msgs else []
                
            for msg in out_msgs:
                if not isinstance(msg, dict):
                    continue
                dest = msg.get('destination', {})
                if not dest:
                    continue
                dest_addr = self._normalize_address(dest.get('address', ''))
                if dest_addr in fee_whitelist and dest_addr not in found_addresses:
                    # Found fee destination in message, record as hit with 0 amount
                    marketplace_type = fee_whitelist[dest_addr]
                    value = int(msg.get('value', 0) or 0)
                    self._record_fee_hit(node, dest_addr, marketplace_type, value, False, lt, within_scope, "message_destination")
                    found_addresses.add(dest_addr)
        
        # Traverse for message-based hits
        UnifiedTraversal.simple_traverse(trace, scan_node_messages)
    
    def _collect_fee_hits_relaxed(self, trace: Dict, sale_contract_addr: str, start_lt: int, end_lt: int):
        """Relaxed fee collection with linkage criteria to avoid cross-batch contamination"""
        fee_whitelist = {self._normalize_address(a): t for a, t in self.marketplace_addresses.items()}
        found_addresses = {hit.address for hit in self.fee_hits}
        
        # Build sale contract linkage index for validation
        by_source_index = self._build_by_source_index(trace)
        sale_contract_children = set()
        normalized_sale_addr = self._normalize_address(sale_contract_addr)
        if normalized_sale_addr in by_source_index:
            for child in by_source_index[normalized_sale_addr]:
                sale_contract_children.add(self._normalize_address(child.get('transaction', {}).get('account', {}).get('address', '')))
        
        def visit_relaxed_node(node):
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
                
            # Check extended LT window (transaction.lt OR in_msg.created_lt)
            lt = tx.get('lt', 0)
            in_msg = tx.get('in_msg', {})
            in_msg_created_lt = in_msg.get('created_lt', 0) if in_msg else 0
            
            # Accept if either transaction LT or message created LT is in window
            tx_in_window = start_lt <= lt <= end_lt
            msg_in_window = start_lt <= in_msg_created_lt <= end_lt if in_msg_created_lt > 0 else False
            
            if not (tx_in_window or msg_in_window):
                return
                
            # Check if this account is a marketplace fee account
            acc_addr = self._normalize_address(tx.get('account', {}).get('address', ''))
            if acc_addr not in fee_whitelist or acc_addr in found_addresses:
                return
                
            # Linkage validation to prevent cross-batch contamination
            linkage = "unknown"
            if in_msg:
                source_addr = self._normalize_address(in_msg.get('source', {}).get('address', ''))
                if source_addr == normalized_sale_addr:
                    linkage = "source=sale_contract"
                elif source_addr in sale_contract_children:
                    linkage = "child_of_sale"
                else:
                    # Skip fees without clear linkage to our sale
                    return
            else:
                # No in_msg - could be internal/aborted, still record if it's a known fee address
                linkage = "no_in_msg"
                
            # Record the fee hit (accept 0 TON with proper linkage)
            credit = int(tx.get('credit_phase', {}).get('credit', 0) or 0)
            value = int(in_msg.get('value', 0) or 0) if in_msg else 0
            aborted = bool(tx.get('aborted', False))
            marketplace_type = fee_whitelist[acc_addr]
            
            # Use value if credit is 0 (common for post-settlement fees)
            effective_amount = credit if credit > 0 else value
            
            # Enhanced linkage with timing info
            timing_info = f"tx_lt={lt}"
            if msg_in_window and not tx_in_window:
                timing_info += f",msg_created_lt={in_msg_created_lt}(post-settlement)"
            enhanced_linkage = f"{linkage}({timing_info})"
            
            self._record_fee_hit(node, acc_addr, marketplace_type, effective_amount, aborted, lt, False, enhanced_linkage)
            found_addresses.add(acc_addr)
        
        # Traverse for relaxed hits
        UnifiedTraversal.simple_traverse(trace, visit_relaxed_node)
        
        # Return collected fee hits
        return self.fee_hits
    
    def _build_fallback_scope(self, trace: Dict, event: Dict) -> AnalysisScope:
        """Build minimal scope for fee collection when NFT-specific scope is not available"""
        root_tx = trace.get('transaction', {})
        root_lt = root_tx.get('lt', 0)
        
        # Create a conservative LT window around the root transaction
        start_lt = root_lt
        end_lt = root_lt + 100000  # Small window to limit scope
        
        # Collect all transaction accounts in this window as allowed nodes
        allowed_accounts = set()
        allowed_node_ids = set()
        
        def collect_nodes_in_window(node):
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
                
            lt = tx.get('lt', 0)
            if start_lt <= lt <= end_lt:
                allowed_node_ids.add(id(node))
                account = self._normalize_address(tx.get('account', {}).get('address', ''))
                if account:
                    allowed_accounts.add(account)
        
        UnifiedTraversal.simple_traverse(trace, collect_nodes_in_window)
        
        self.logger.debug(f"   🔧 Built fallback scope: {len(allowed_node_ids)} nodes, LT: {start_lt}-{end_lt}")
        
        return AnalysisScope(
            allowed_node_ids=allowed_node_ids,
            start_lt=start_lt,
            end_lt=end_lt,
            sale_contract_addr="",  # Unknown for fallback
            nft_address=""  # Will be filled by caller if needed
        )
    
    def _handle_out_of_scope_fees(self, fee_breakdown: Dict[str, float], fee_hits: List[FeeHit], marketplace_fees: List[NetFlow], sale_amount: float = 0.0, seller: str = "") -> None:
        """Handle out-of-scope fees for CSV consistency - ONLY when valid sale exists"""
        # GATED: Only process out-of-scope fees if we have a valid sale
        if not seller or sale_amount <= 0:
            self.logger.debug(f"   🚫 Skipping out-of-scope fees: no valid sale (seller={bool(seller)}, amount={sale_amount})")
            return
        
        # Only consider within_scope fee hits to prevent amount mutations from unrelated transactions
        for marketplace_type in ['fragment', 'marketapp', 'getgems']:
            if fee_breakdown.get(marketplace_type, 0) == 0:
                # STRICT: Only within_scope hits allowed, BUT allow Fragment fees in relaxed window
                relevant_hits = [h for h in fee_hits if h.marketplace_type == marketplace_type and (h.within_scope or (marketplace_type == 'fragment' and 'child_of_sale' in h.linkage))]
                if len(relevant_hits) == 1:  # Clear single hit
                    hit = relevant_hits[0]
                    fee_breakdown[marketplace_type] = hit.amount_ton
                    self.logger.debug(f"   🔄 Within-scope fee included: {marketplace_type} = {hit.amount_ton:.3f} TON")
    
    def _normalize_address(self, addr: str) -> str:
        """Normalize address for consistent comparison"""
        if not addr:
            return ""
        
        # Convert to lowercase and strip whitespace
        normalized = addr.lower().strip()
        
        # Ensure workchain prefix format
        if ':' not in normalized and len(normalized) >= 40:
            # Add workchain 0: prefix if missing
            normalized = f"0:{normalized}"
        
        return normalized
    
    def _address_in_list(self, target_addr: str, address_list: List[str]) -> bool:
        """Check if address is in list using are_addresses_equal"""
        return any(are_addresses_equal(target_addr, addr) for addr in address_list)
    
    def _find_topmost_external_wallet(self, event: Dict, trace: Dict) -> str:
        """
        Find topmost external wallet - robust implementation.
        1) Check root for wallet patterns
        2) Search tree for external wallets
        3) Exclude sale/nft/marketplace addresses
        """
        # Build exclusion set
        scope = getattr(self, '_current_analysis_scope', None)
        sale_contract = getattr(self, '_last_sale_contract_addr', None) or (scope.sale_contract_addr if scope else None)
        nft_address = getattr(self, '_current_nft_address', None) or (scope.nft_address if scope else None)
        
        exclusions = set()
        if sale_contract:
            exclusions.add(self._normalize_address(sale_contract))
        if nft_address:
            exclusions.add(self._normalize_address(nft_address))
        # Add all marketplace addresses
        exclusions.update({self._normalize_address(a) for a in getattr(self, 'marketplace_addresses', {}).keys()})
        
        def is_external_wallet(node):
            """Check if node is an external wallet transaction"""
            if not isinstance(node, dict):
                return False
                
            tx = node.get("transaction", {})
            if not isinstance(tx, dict):
                return False
                
            acc_addr = tx.get("account", {}).get("address", "")
            if not acc_addr:
                return False
                
            # Check exclusions
            if self._normalize_address(acc_addr) in exclusions:
                return False
                
            # Check for wallet indicators
            in_msg = tx.get("in_msg", {}) or {}
            op_name = in_msg.get("decoded_op_name", "") or ""
            msg_type = in_msg.get("msg_type", "")
            
            # Strong wallet indicators
            wallet_ops = {
                "wallet_signed_external", "wallet_signed_external_v3",
                "wallet_signed_external_v4", "wallet_signed_external_v5_r1"
            }
            
            if op_name in wallet_ops or msg_type == "external_in":
                return True
                
            # Check for wallet patterns in decoded body
            decoded_body = in_msg.get("decoded_body", {}) or {}
            if "seqno" in decoded_body or "subwallet_id" in decoded_body:
                return True
                
            # Check for simple_send pattern
            payload = decoded_body.get("payload", {})
            if isinstance(payload, dict) and "simple_send" in payload:
                return True
                
            return False
        
        # 1) Check root first
        if is_external_wallet(trace):
            root_addr = trace.get("transaction", {}).get("account", {}).get("address", "")
            self.logger.debug(f"   👤 Found root wallet: {root_addr[-20:]}")
            return root_addr
        
        # 2) Search tree for external wallet (BFS for topmost)
        candidates = []
        
        def search_tree(node, depth=0):
            if is_external_wallet(node):
                addr = node.get("transaction", {}).get("account", {}).get("address", "")
                candidates.append((depth, addr))
                
            # Check children
            children = node.get("children", []) if isinstance(node, dict) else []
            for child in children:
                if isinstance(child, dict):
                    search_tree(child, depth + 1)
        
        search_tree(trace)
        
        if candidates:
            # Return topmost (lowest depth) wallet
            candidates.sort(key=lambda x: x[0])
            best_wallet = candidates[0][1]
            self.logger.debug(f"   👤 Found wallet at depth {candidates[0][0]}: {best_wallet[-20:]}")
            return best_wallet
        
        # 3) Last resort: root if not excluded
        root_addr = trace.get("transaction", {}).get("account", {}).get("address", "")
        if root_addr and self._normalize_address(root_addr) not in exclusions:
            self.logger.debug(f"   👤 Fallback to root (not excluded): {root_addr[-20:]}")
            return root_addr
        
        self.logger.debug(f"   👤 No valid wallet found")
        return ""
        if nft_address:
            exclusions.add(self._normalize_address(nft_address))
        # Add marketplace addresses
        exclusions.update({self._normalize_address(a) for a in getattr(self, 'marketplace_addresses', {}).keys()})
        
        # Check if root is valid wallet (not excluded)
        root_normalized = self._normalize_address(root_acct)
        
        if is_wallet_external and root_normalized not in exclusions:
            self.logger.debug(f"   👤 Root wallet found: {root_acct[-20:]} (op: {op_name or 'external_in'})")
            return root_acct
        
        # Log why root was rejected
        if root_normalized in exclusions:
            reason = "sale_contract" if sale_contract and root_normalized == self._normalize_address(sale_contract) else \
                     "nft" if nft_address and root_normalized == self._normalize_address(nft_address) else \
                     "marketplace"
            self.logger.debug(f"   👤 Root {root_acct[-20:]} excluded: {reason}")
        elif not is_wallet_external:
            self.logger.debug(f"   👤 Root {root_acct[-20:]} not a wallet op: {op_name}")
        
        # 1-hop fallback: check parent node
        parent = trace.get("parent", {})
        if isinstance(parent, dict):
            parent_tx = parent.get("transaction", {}) or {}
            parent_acct = parent_tx.get("account", {}).get("address", "") or ""
            
            if parent_acct and self._normalize_address(parent_acct) not in exclusions:
                self.logger.debug(f"   👤 Parent wallet found: {parent_acct[-20:]}")
                return parent_acct
        
        # Last resort: return root anyway (guards will handle it later)
        self.logger.debug(f"   👤 Fallback to root (guards will filter): {root_acct[-20:]}")
        return root_acct or ""
    
    def _build_by_dest_index(self, trace: Dict) -> Dict[str, List[str]]:
        """Build index of destination -> list of senders"""
        by_dest = {}
        
        def _traverse_for_index(node: Dict):
            tx = node.get('transaction', {})
            account_addr = tx.get('account', {}).get('address', '')
            
            # Check outgoing messages
            for out_msg in tx.get('out_msgs', []):
                dest_addr = out_msg.get('destination', {}).get('address', '')
                if dest_addr:
                    if dest_addr not in by_dest:
                        by_dest[dest_addr] = []
                    if account_addr and account_addr not in by_dest[dest_addr]:
                        by_dest[dest_addr].append(account_addr)
            
            # Recurse to children
            for child in node.get('children', []):
                _traverse_for_index(child)
        
        _traverse_for_index(trace)
        return by_dest
    
    def _extract_all_nodes(self, trace: Dict) -> List[str]:
        """Extract all unique node addresses from trace"""
        all_nodes = set()
        
        def _traverse_for_nodes(node: Dict):
            tx = node.get('transaction', {})
            account_addr = tx.get('account', {}).get('address', '')
            if account_addr:
                all_nodes.add(account_addr)
            
            # Recurse to children
            for child in node.get('children', []):
                _traverse_for_nodes(child)
        
        _traverse_for_nodes(trace)
        return list(all_nodes)
    
    def _is_descendant_of_sale(self, node: Dict, sale_contract_addr: str, start_lt: int, end_lt: int) -> bool:
        """Check if node is a direct or grandchild descendant of sale contract within LT window"""
        tx = node.get('transaction', {})
        if not tx:
            return False
            
        # Check LT window first
        lt = tx.get('lt', 0)
        if not (start_lt <= lt <= end_lt):
            return False
        
        # Get node address
        node_addr = self._normalize_address(tx.get('account', {}).get('address', ''))
        if not node_addr:
            return False
        
        # Use both routing indices
        by_source = getattr(self, '_by_source_index', {})
        by_dest = getattr(self, '_by_dest_index', {})
        sale_norm = self._normalize_address(sale_contract_addr)
        
        # Check direct children
        direct_children = by_source.get(sale_norm, []) + by_dest.get(sale_norm, [])
        for child in direct_children:
            child_addr = self._normalize_address(
                child.get('transaction', {}).get('account', {}).get('address', '')
            )
            if child_addr == node_addr:
                return True
        
        # Check grandchildren (one level deeper)
        for child in direct_children:
            child_addr = self._normalize_address(
                child.get('transaction', {}).get('account', {}).get('address', '')
            )
            grandchildren = by_source.get(child_addr, []) + by_dest.get(child_addr, [])
            for grandchild in grandchildren:
                gc_addr = self._normalize_address(
                    grandchild.get('transaction', {}).get('account', {}).get('address', '')
                )
                if gc_addr == node_addr:
                    return True
        
        return False
    
    def _build_expanded_lt_window(self, trace: Dict, sale_contract_addr: str, start_lt: int, end_lt: int, 
                                 by_source_index: Dict, by_dest_index: Dict) -> tuple:
        """Build dynamically expanded LT window based on graph neighbors of sale contract"""
        sale_norm = self._normalize_address(sale_contract_addr)
        
        # Collect LT values of direct neighbors (parents + children)
        neighbor_lts = []
        
        # Direct children of sale contract
        for child in by_source_index.get(sale_norm, []) + by_dest_index.get(sale_norm, []):
            child_lt = child.get('transaction', {}).get('lt', 0)
            if child_lt > 0:
                neighbor_lts.append(child_lt)
        
        # Direct parents of sale contract (nodes that send to it)
        for source_addr, nodes in by_source_index.items():
            for node in nodes:
                for out_msg in node.get('transaction', {}).get('out_msgs', []):
                    dest_addr = self._normalize_address(
                        out_msg.get('destination', {}).get('address', '')
                    )
                    if dest_addr == sale_norm:
                        parent_lt = node.get('transaction', {}).get('lt', 0)
                        if parent_lt > 0:
                            neighbor_lts.append(parent_lt)
        
        # Calculate expanded window
        if neighbor_lts:
            expanded_start_lt = min(start_lt, min(neighbor_lts))
            expanded_end_lt = max(end_lt, max(neighbor_lts))
        else:
            # Fallback: minimal expansion
            expanded_start_lt = start_lt - 3
            expanded_end_lt = end_lt + 3
        
        self.logger.debug(f"   🛟 PRE-PASS: Expanded LT window [{expanded_start_lt}, {expanded_end_lt}] "
                         f"from [{start_lt}, {end_lt}] via {len(neighbor_lts)} neighbors")
        
        return expanded_start_lt, expanded_end_lt
    
    def _allow_top_credit_candidate_strict(self, trace: Dict, start_lt: int, end_lt: int, 
                                          remaining_budget: Dict, exclude_ids: set,
                                          sale_contract_addr: str, nft_address: str) -> Optional[int]:
        """Strict seller-rescue candidate finder with utime clamp and graph linkage"""
        target = abs(remaining_budget['amount'])
        min_credit_ratio = 0.6  # Require at least 60% of remaining budget
        min_credit = min_credit_ratio * target
        best = (None, float('inf'))
        
        # Get sale contract utime for time clamping
        sale_contract_utime = 0
        def find_sale_utime(node):
            nonlocal sale_contract_utime
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
            addr = self._normalize_address(tx.get('account', {}).get('address', ''))
            if addr == self._normalize_address(sale_contract_addr):
                sale_contract_utime = tx.get('now', 0)
        
        UnifiedTraversal.simple_traverse(trace, find_sale_utime)
        
        def scan_strict(node):
            nonlocal best
            if not isinstance(node, dict):
                return
            
            tx = node.get('transaction', {})
            if not tx:
                return
                
            # Check LT window (expanded)
            lt = tx.get('lt', 0)
            addr = tx.get('account', {}).get('address', '')
            if not (start_lt <= lt <= end_lt):
                return
                
            # Utime clamp: ±45 seconds from sale contract
            node_utime = tx.get('now', 0)
            if sale_contract_utime > 0 and abs(node_utime - sale_contract_utime) > 45:
                self.logger.debug(f"   🔍 STRICT: {addr[-12:]} utime Δ={abs(node_utime - sale_contract_utime)}s > 45s")
                return
                
            # Skip already allowed nodes
            node_id = id(node)
            if node_id in exclude_ids:
                return
                
            # Skip sale contract, NFT contract, fee addresses
            addr_norm = self._normalize_address(addr)
            if (addr_norm == self._normalize_address(sale_contract_addr) or
                addr_norm == self._normalize_address(nft_address) or
                addr_norm in {self._normalize_address(a) for a in self.marketplace_addresses}):
                return
                
            # Check for positive credit
            credit_phase = tx.get('credit_phase', {})
            if 'credit' not in credit_phase:
                return
                
            try:
                credit_amount = int(credit_phase['credit'])
            except (ValueError, TypeError):
                return
                
            if credit_amount < min_credit:
                self.logger.debug(f"   🔍 STRICT: {addr[-12:]} credit={credit_amount/1e9:.3f} < min_credit={min_credit/1e9:.3f}")
                return
                
            # Check graph linkage (is descendant of sale contract)
            if not self._is_descendant_of_sale_simple(node, sale_contract_addr, start_lt, end_lt):
                self.logger.debug(f"   🔍 STRICT: {addr[-12:]} not linked to sale contract")
                return
            
            # NEW: CRITICAL CHECK - Does this node send money to OUR NFT?
            # A valid seller must either:
            # 1. Send money directly to our NFT address
            # 2. Be part of a chain that leads to our NFT
            
            sends_to_our_nft = False
            out_msgs = tx.get('out_msgs', [])
            if not isinstance(out_msgs, list):
                out_msgs = [out_msgs] if out_msgs else []
            
            nft_normalized = self._normalize_address(nft_address)
            
            # Check direct sends to NFT
            for msg in out_msgs:
                if isinstance(msg, dict):
                    dest = msg.get('destination', {}).get('address', '')
                    if dest and self._normalize_address(dest) == nft_normalized:
                        sends_to_our_nft = True
                        self.logger.debug(f"   ✅ STRICT: {addr[-12:]} sends to OUR NFT")
                        break
            
            # If not direct, check if sale contract sends to our NFT
            # (the chain must be: seller -> sale_contract -> our_nft)
            if not sends_to_our_nft:
                # Check if our sale contract sends to our NFT
                sale_sends_to_nft = self._sale_contract_sends_to_nft(
                    trace, sale_contract_addr, nft_address
                )
                if not sale_sends_to_nft:
                    self.logger.debug(f"   🚫 STRICT: Sale contract doesn't send to our NFT - wrong tree!")
                    return
                
            # Calculate difference from remaining budget
            diff = abs(credit_amount - target)
            tolerance = max(1e9, 0.05 * max(target, 1))  # 5% tolerance
            
            utime_delta = abs(node_utime - sale_contract_utime) if sale_contract_utime > 0 else 0
            self.logger.debug(f"   🛟 RESCUE CANDIDATE: {addr[-12:]} credit={credit_amount/1e9:.3f} TON, "
                             f"diff={diff/1e9:.3f}, utime Δ={utime_delta}s, linked=True")
            
            if diff <= tolerance and diff < best[1]:
                best = (node_id, diff)
        
        # Scan entire trace for strict candidates
        UnifiedTraversal.simple_traverse(trace, scan_strict)
        
        if best[0] is not None:
            self.logger.debug(f"   ✅ RESCUE: Selected strict seller-rescue candidate (diff: {best[1]/1e9:.3f} TON)")
        
        return best[0]
    
    def _sale_contract_sends_to_nft(self, trace: Dict, sale_contract_addr: str, nft_address: str) -> bool:
        """Check if sale contract sends money to the specific NFT"""
        sale_node = self._find_node_by_address(trace, sale_contract_addr)
        if not sale_node:
            return False
        
        tx = sale_node.get('transaction', {})
        out_msgs = tx.get('out_msgs', [])
        if not isinstance(out_msgs, list):
            out_msgs = [out_msgs] if out_msgs else []
        
        nft_normalized = self._normalize_address(nft_address)
        
        for msg in out_msgs:
            if isinstance(msg, dict):
                dest = msg.get('destination', {}).get('address', '')
                if dest and self._normalize_address(dest) == nft_normalized:
                    return True
        
        return False
    
    def _is_descendant_of_sale_simple(self, node: Dict, sale_contract_addr: str, start_lt: int, end_lt: int) -> bool:
        """Simple BFS to check if node is descendant of sale contract within LT window"""
        target_id = id(node)
        sale_norm = self._normalize_address(sale_contract_addr)
        
        # BFS from sale contract
        visited = set()
        queue = []
        
        # Start with sale contract's children (use stored indices with fallback)
        by_source = getattr(self, '_by_source_index', {}) or {}
        by_dest = getattr(self, '_by_dest_index', {}) or {}
        
        for child in by_source.get(sale_norm, []) + by_dest.get(sale_norm, []):
            child_lt = child.get('transaction', {}).get('lt', 0)
            if start_lt <= child_lt <= end_lt:
                queue.append(child)
        
        while queue:
            current = queue.pop(0)
            current_id = id(current)
            
            if current_id == target_id:
                return True
                
            if current_id in visited:
                continue
            visited.add(current_id)
            
            # Add children of current node
            current_addr = self._normalize_address(
                current.get('transaction', {}).get('account', {}).get('address', '')
            )
            for child in by_source.get(current_addr, []) + by_dest.get(current_addr, []):
                child_lt = child.get('transaction', {}).get('lt', 0)
                if start_lt <= child_lt <= end_lt and id(child) not in visited:
                    queue.append(child)
        
        return False
    
    def _score_sale_contract_candidates(self, trace: Dict, candidates: set, nft_seeds: List[Dict], nft_address: str) -> str:
        """Score sale contract candidates and return the best one"""
        if not candidates:
            return ""
        
        candidate_scores = {}
        
        # Get NFT credit LT for temporal scoring
        nft_credit_lt = 0
        for seed in nft_seeds:
            lt = seed.get('transaction', {}).get('lt', 0)
            if lt > nft_credit_lt:
                nft_credit_lt = lt
        
        for candidate in candidates:
            score = 0
            candidate_normalized = self._normalize_address(candidate)
            
            # Score 1: Known marketplace contract (+2)
            if candidate in self.marketplace_addresses:
                score += 2
                self.logger.debug(f"   📊 {candidate[-12:]}: +2 (marketplace contract)")
            
            # Score 2: Temporal proximity to NFT credit (+1 for closest)
            candidate_node = self._find_node_by_address(trace, candidate)
            if candidate_node:
                candidate_lt = candidate_node.get('transaction', {}).get('lt', 0)
                if candidate_lt > 0 and nft_credit_lt > 0:
                    # Closer to NFT credit LT is better
                    lt_distance = abs(candidate_lt - nft_credit_lt)
                    if lt_distance <= 5:  # Very close
                        score += 2
                        self.logger.debug(f"   📊 {candidate[-12:]}: +2 (very close LT: {lt_distance})")
                    elif lt_distance <= 20:  # Reasonably close
                        score += 1
                        self.logger.debug(f"   📊 {candidate[-12:]}: +1 (close LT: {lt_distance})")
            
            # Score 3: Has out_msgs to relevant addresses (+1)
            if self._has_relevant_outgoing_messages(trace, candidate, nft_address):
                score += 1
                self.logger.debug(f"   📊 {candidate[-12:]}: +1 (relevant out_msgs)")
            
            candidate_scores[candidate] = score
            self.logger.debug(f"   📊 {candidate[-12:]}: total score = {score}")
        
        # Return highest scoring candidate
        if candidate_scores:
            best_candidate = max(candidate_scores, key=candidate_scores.get)
            self.logger.debug(f"   🏆 Best candidate: {best_candidate[-12:]} (score: {candidate_scores[best_candidate]})")
            return best_candidate
        
        return ""
    
    def _has_relevant_outgoing_messages(self, trace: Dict, candidate_addr: str, nft_address: str) -> bool:
        """Check if candidate has outgoing messages to NFT or fee addresses"""
        candidate_node = self._find_node_by_address(trace, candidate_addr)
        if not candidate_node:
            return False
        
        tx = candidate_node.get('transaction', {})
        out_msgs = tx.get('out_msgs', [])
        if not isinstance(out_msgs, list):
            out_msgs = [out_msgs] if out_msgs else []
        
        nft_normalized = self._normalize_address(nft_address)
        marketplace_addresses = {self._normalize_address(addr) for addr in self.marketplace_addresses.keys()}
        
        for msg in out_msgs:
            if not isinstance(msg, dict):
                continue
            dest_addr = msg.get('destination', {}).get('address', '') or msg.get('dest', '')
            if dest_addr:
                dest_normalized = self._normalize_address(dest_addr)
                if dest_normalized == nft_normalized or dest_normalized in marketplace_addresses:
                    return True
        
        return False
    
    def _is_child_of_sale_contract(self, node: Dict, sale_contract_addr: str, fee_addr: str) -> bool:
        """Check if fee address receives out_msg from sale contract (Fragment child transactions)"""
        try:
            # Find sale contract node in trace
            def find_sale_node(trace_node):
                if not isinstance(trace_node, dict):
                    return None
                tx = trace_node.get('transaction', {})
                account_addr = tx.get('account', {}).get('address', '')
                if are_addresses_equal(account_addr, sale_contract_addr):
                    return trace_node
                for child in trace_node.get('children', []):
                    result = find_sale_node(child)
                    if result:
                        return result
                return None
            
            # This is a simplified check - in practice you'd traverse the full trace
            # For now, just check if addresses are related in the context
            return False  # Placeholder - implement full out_msg check if needed
            
        except Exception as e:
            self.logger.debug(f"   ❌ Child-of-sale check error: {e}")
            return False
    
    def _setup_nft_specific_fee_context(self, trace: Dict, sale_contract_addr: str, nft_address: str, start_lt: int, end_lt: int):
        """Set up context for NFT-specific fee collection with tight source binding"""
        # Find key LT timestamps for this NFT instance
        sale_contract_lt = None
        nft_credit_lt = None
        seller_payout_lt = None
        
        # Find sale contract LT
        def find_sale_lt(node):
            nonlocal sale_contract_lt
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
            addr = self._normalize_address(tx.get('account', {}).get('address', ''))
            if addr == self._normalize_address(sale_contract_addr):
                credit = int(tx.get('credit_phase', {}).get('credit', 0) or 0)
                if credit > 1e8:  # Significant credit (sale amount)
                    sale_contract_lt = tx.get('lt', 0)
        
        # Find NFT contract credit LT
        def find_nft_credit_lt(node):
            nonlocal nft_credit_lt
            if not isinstance(node, dict):
                return
            tx = node.get('transaction', {})
            if not tx:
                return
            addr = self._normalize_address(tx.get('account', {}).get('address', ''))
            if addr == self._normalize_address(nft_address):
                credit = int(tx.get('credit_phase', {}).get('credit', 0) or 0)
                if credit > 0:  # Any NFT contract credit
                    nft_credit_lt = tx.get('lt', 0)
        
        # Find seller payout LT (from seller-rescue if available)
        if self.state.has_seller_credits:
            def find_seller_payout_lt(node):
                nonlocal seller_payout_lt
                if not isinstance(node, dict):
                    return
                tx = node.get('transaction', {})
                if not tx:
                    return
                credit = int(tx.get('credit_phase', {}).get('credit', 0) or 0)
                # Look for significant seller credit (close to sale amount)
                if credit > 1e8:  # > 0.1 TON
                    addr = self._normalize_address(tx.get('account', {}).get('address', ''))
                    # Exclude sale contract and NFT contract
                    if (addr != self._normalize_address(sale_contract_addr) and 
                        addr != self._normalize_address(nft_address)):
                        seller_payout_lt = tx.get('lt', 0)
            
            UnifiedTraversal.simple_traverse(trace, find_seller_payout_lt)
        
        # Scan trace for LT values
        UnifiedTraversal.simple_traverse(trace, find_sale_lt)
        UnifiedTraversal.simple_traverse(trace, find_nft_credit_lt)
        
        # Use provided LT bounds if specific LTs not found
        if not sale_contract_lt:
            sale_contract_lt = start_lt
        if not nft_credit_lt:
            nft_credit_lt = end_lt
            
        # Set up the context
        self._nft_specific_fee_context = {
            'sale_contract_addr': sale_contract_addr,
            'nft_address': nft_address,
            'sale_lt': sale_contract_lt,
            'nft_credit_lt': nft_credit_lt,
            'seller_payout_lt': seller_payout_lt,
            'start_lt': start_lt,
            'end_lt': end_lt
        }
        
        self.logger.debug(f"   🎯 NFT-SPECIFIC CONTEXT: sale_lt={sale_contract_lt}, nft_credit_lt={nft_credit_lt}, "
                         f"seller_payout_lt={seller_payout_lt}, sale_contract={sale_contract_addr[-12:]}")
    
    def _aggregate_value_flows(self, event: Dict, trace: Dict, nft_address: str = None) -> List[NetFlow]:
        """Aggregate value flows by address - NFT-specific if nft_address provided"""
        from .config import nanoton_to_ton, TON_PRECISION_CALCULATION
        
        # Collect all value flows
        all_flows = []
        
        if nft_address:
            # NFT-SPECIFIC: Extract from raw trace transaction details
            trace_flows = self._extract_nft_specific_flows_from_trace(trace, nft_address)
            all_flows.extend(trace_flows)
            self.logger.debug(f"   🔍 Extracted {len(trace_flows)} flows from raw trace for {nft_address[-10:]}")
        else:
            # FALLBACK: Use event flows (legacy behavior)
            if event and 'value_flow' in event:
                all_flows.extend(event['value_flow'])
            
            if trace and 'value_flow' in trace:
                all_flows.extend(trace['value_flow'])
        
        # Aggregate by address
        address_aggregates = {}
        
        for flow in all_flows:
            if not isinstance(flow, dict):
                continue
                
            account = flow.get('account', {})
            if not isinstance(account, dict):
                continue
                
            address = account.get('address', '')
            if not address:
                continue
            
            # Convert nanoTON to TON
            flow_amount_nanoton = flow.get('amount_nanoton', 0)
            try:
                flow_amount_ton = nanoton_to_ton(int(flow_amount_nanoton), TON_PRECISION_CALCULATION)
            except (ValueError, TypeError):
                continue
            
            # Skip dust amounts (EXCEPT for marketplace addresses - they're always relevant)
            is_marketplace_addr = address in self.marketplace_addresses
            if not is_marketplace_addr and abs(flow_amount_ton) < self.MIN_SIGNIFICANT_AMOUNT_TON:
                continue
            
            # Initialize or update aggregate
            if address not in address_aggregates:
                address_aggregates[address] = {
                    'total_in': 0.0,
                    'total_out': 0.0,
                    'net': 0.0
                }
            
            if flow_amount_ton > 0:
                address_aggregates[address]['total_in'] += flow_amount_ton
            else:
                address_aggregates[address]['total_out'] += abs(flow_amount_ton)
            
            address_aggregates[address]['net'] += flow_amount_ton
        
        # Convert to NetFlow objects
        net_flows = []
        for address, agg in address_aggregates.items():
            is_marketplace = address in self.marketplace_addresses
            marketplace_type = self.marketplace_addresses.get(address, "") if is_marketplace else ""
            
            # Debug address matching for known marketplace addresses
            from .config import MARKETAPP_MARKETPLACE_ADDRESS, GETGEMS_SALES_ADDRESS, GETGEMS_FEE_ADDRESS
            if address in [MARKETAPP_MARKETPLACE_ADDRESS, GETGEMS_SALES_ADDRESS, GETGEMS_FEE_ADDRESS]:
                self.logger.debug(f"   🔍 MARKETPLACE CHECK: {address[-20:]} → is_marketplace={is_marketplace}, type={marketplace_type}")
                if not is_marketplace:
                    self.logger.debug(f"   ❌ Address not in marketplace_addresses. Available: {list(self.marketplace_addresses.keys())}")
            
            # Collect flow types for this address and exclude buyer_change from net calculation
            address_flow_types = []
            buyer_change_amount = 0.0
            for flow in all_flows:
                flow_addr = flow.get('account', {}).get('address', '')
                if flow_addr == address and 'flow_type' in flow:
                    flow_type = flow['flow_type']
                    address_flow_types.append(flow_type)
                    
                    # Exclude buyer_change from net calculation
                    if flow_type == 'buyer_change':
                        try:
                            buyer_change_amount += nanoton_to_ton(int(flow.get('amount_nanoton', 0)), TON_PRECISION_CALCULATION)
                        except (ValueError, TypeError):
                            pass
            
            # Adjust net amount to exclude buyer change
            adjusted_net = agg['net'] - buyer_change_amount
            if buyer_change_amount != 0:
                self.logger.debug(f"   🔁 Excluded buyer_change from {address[-10:]}: {buyer_change_amount:.6f} TON (net: {agg['net']:.6f} → {adjusted_net:.6f})")
            
            net_flow = NetFlow(
                address=address,
                net_amount_ton=adjusted_net,
                total_in_ton=agg['total_in'],
                total_out_ton=agg['total_out'],
                is_marketplace_fee_account=is_marketplace,
                marketplace_type=marketplace_type,
                flow_types=list(set(address_flow_types))  # Unique flow types
            )
            net_flows.append(net_flow)
        
        return net_flows
    
    def _identify_marketplace_fees(self, net_flows: List[NetFlow]) -> List[NetFlow]:
        """Identify marketplace fee flows from whitelist"""
        marketplace_fees = []
        
        for flow in net_flows:
            if flow.is_marketplace_fee_account:  # Accept any amount - presence indicates marketplace involvement
                marketplace_fees.append(flow)
        
        return marketplace_fees
    
    def _pick_seller_from_path(self, fallback_seller: str) -> str:
        """Choose seller strictly from direct children of sale contract (no cross-tree widening)."""
        
        # Get path flows for seller detection
        path_flows = getattr(self, '_last_path_flows', []) or []
        
        # Check if we have any path flows at all
        if not path_flows:
            self.logger.debug("   ⚠️ No path flows available")
            return fallback_seller if fallback_seller else ""
        
        # Aggregate seller credits from path flows
        from collections import defaultdict
        path_seller_credits = defaultdict(int)
        
        for f in path_flows:
            flow_type = f.get('flow_type', '')
            amount_nanoton = f.get('amount_nanoton', 0)
            
            # Check for seller credit flows
            if flow_type in ('seller_credit', 'seller_credit_inferred') and amount_nanoton > 0:
                account_dict = f.get('account', {})
                if isinstance(account_dict, dict):
                    addr = account_dict.get('address', '')
                else:
                    addr = str(account_dict) if account_dict else ''
                
                if addr and addr != 'teleitem_bid_inferred':  # Exclude synthetic addresses
                    addr_norm = self._normalize_address(addr)
                    
                    # Exclude buyer and marketplace addresses
                    if (addr_norm == self._normalize_address(getattr(self, '_detected_buyer_wallet', '')) or
                        self._is_marketplace_contract(addr)):
                        continue
                        
                    path_seller_credits[addr_norm] += amount_nanoton
        
        # If path seller found, return immediately
        if path_seller_credits:
            best_addr = max(path_seller_credits.items(), key=lambda kv: kv[1])[0]
            self.logger.debug(f"   🎯 PATH SELLER (final): {best_addr[-10:]} = {path_seller_credits[best_addr]/1e9:.3f} TON")
            return best_addr
        
        # No seller credits found - log details for debugging
        self.logger.debug(f"   ⚠️ No seller credits found in {len(path_flows)} path flows")
        
        # Debug what flow types we actually have
        flow_types_found = [f.get('flow_type', 'unknown') for f in path_flows]
        if flow_types_found:
            self.logger.debug(f"   🔍 Flow types in path: {flow_types_found}")
        
        return fallback_seller if fallback_seller else ""
    
    def _is_on_nft_path(self, address: str) -> bool:
        """Check if address appears in the extracted NFT path flows."""
        path_flows = getattr(self, '_last_path_flows', []) or []
        return any(self._normalize_address((f.get('account') or {}).get('address','')) ==
                   self._normalize_address(address) for f in path_flows)
    
    def _determine_seller_buyer(self, net_flows: List[NetFlow], marketplace_fees: List[NetFlow]) -> Tuple[str, str, float]:
        """
        Determine seller and buyer using largest net flows
        
        Seller = largest positive net flow (excluding marketplace fees, NFT address, sale contract)
        Buyer = largest negative net flow (excluding NFT address, sale contract)
        Sale amount = seller's positive net flow
        """
        marketplace_addresses = {fee.address for fee in marketplace_fees}
        
        # Get NFT address and sale contract address from current analysis scope
        current_scope = getattr(self, '_current_analysis_scope', None)
        nft_address = current_scope.nft_address if current_scope else ''
        sale_contract_addr = current_scope.sale_contract_addr if current_scope else ''
        
        # Global blacklist: marketplace fees, NFT contract, sale contract
        excluded_addresses = marketplace_addresses.copy()
        if nft_address:
            excluded_addresses.add(self._normalize_address(nft_address))
        if sale_contract_addr:
            excluded_addresses.add(self._normalize_address(sale_contract_addr))
        
        # Add all known marketplace addresses to blacklist
        for addr in self.marketplace_addresses.keys():
            excluded_addresses.add(self._normalize_address(addr))
        
        # NEW: Restrict seller candidates to same downstream subtree
        path_addr_set = {self._normalize_address(n.get("address", "")) for n in (self.state.current_analysis_scope.path_nodes or []) if n.get("address")} if self.state.current_analysis_scope else set()
        
        def _is_same_subtree(addr: str) -> bool:
            a = self._normalize_address(addr)
            return (a in path_addr_set)
        
        # Guards against Buyer/Sale/NFT
        detected_buyer = getattr(self, '_detected_buyer_wallet', '')
        guards = {
            self._normalize_address(detected_buyer or ''),
            self._normalize_address(self.state.current_analysis_scope.sale_contract_addr) if self.state.current_analysis_scope else '',
            self._normalize_address(nft_address or '')
        }
        guards.discard('')  # Remove empty strings
        
        # Pre-filter candidates: same subtree + not in guards + above threshold + not marketplace
        initial_candidates = [
            f for f in net_flows
            if (f.net_amount_ton > self.MIN_SIGNIFICANT_AMOUNT_TON
                and _is_same_subtree(f.address)
                and self._normalize_address(f.address) not in guards
                and not self._is_marketplace_contract(f.address))
        ]
        
        self.logger.debug(f"   🔍 Pre-filtered to {len(initial_candidates)} candidates in same subtree")
        self.logger.debug("   🧑‍💼 SELLER CANDIDATES (same subtree): " + ", ".join(
            f"{c.address[-10:]}={c.net_amount_ton:.3f}" for c in initial_candidates))
        
        # --- NEW: Path-only seller selection (no widening beyond NFT path) ---
        from collections import defaultdict
        
        path_seller_credits = defaultdict(int)
        on_path_addresses = set()

        # Only consider flows from the validated NFT path
        for f in getattr(self, '_last_path_flows', []):
            addr = self._normalize_address(f.get('account', {}).get('address', ''))
            on_path_addresses.add(addr)
            if f.get('flow_type') in ('seller_credit', 'seller_credit_inferred') and f.get('amount_nanoton', 0) > 0:
                # Never count buyer change as seller credit
                if addr == self._normalize_address(getattr(self, '_detected_buyer_wallet', '')):
                    continue
                path_seller_credits[addr] += f['amount_nanoton']
        
        # Apply guards to path seller credits
        scope = getattr(self, '_current_analysis_scope', None)
        for guard in [
            getattr(scope, 'sale_contract_addr', ''),
            getattr(scope, 'nft_address', ''),
            *(self.marketplace_addresses.keys()),
            getattr(self, '_detected_buyer_wallet', '')
        ]:
            path_seller_credits.pop(self._normalize_address(guard or ''), None)
        
        if path_seller_credits:
            # Largest positive amount on THE path determines the seller
            seller_addr, seller_nanoton = max(path_seller_credits.items(), key=lambda kv: kv[1])
            seller = seller_addr
            seller_credit_sum = sum(path_seller_credits.values()) / 1e9  # Convert to TON
            self.logger.debug(f"   🎯 PATH SELLER: {seller[-10:]} = {seller_nanoton/1e9:.3f} TON (sum: {seller_credit_sum:.3f})")
            
            # Build seller_candidates from path flows for compatibility
            seller_candidates = [
                type('NetFlow', (), {
                    'address': addr, 
                    'net_amount_ton': amount / 1e9,
                    'flow_types': ['seller_credit']
                })()
                for addr, amount in path_seller_credits.items()
            ]
        else:
            # No seller_credit on path → fallback to transfer signals
            seller = ""
            seller_credit_sum = 0.0
            seller_candidates = []
            
            # Try transfer result fallback
            transfer_result = getattr(self, '_last_transfer_result', None)
            if transfer_result and getattr(transfer_result, 'from_address', ''):
                cand = transfer_result.from_address
                cand_n = self._normalize_address(cand)
                guard_addrs = {
                    self._normalize_address(getattr(scope, 'sale_contract_addr', '')),
                    self._normalize_address(getattr(scope, 'nft_address', '')),
                    *(self._normalize_address(a) for a in self.marketplace_addresses.keys()),
                    self._normalize_address(getattr(self, '_detected_buyer_wallet', ''))
                }
                if cand_n not in guard_addrs:
                    seller = cand
                    self.logger.debug(f"   🔄 TRANSFER FALLBACK SELLER: {seller[-10:]}")
        
        marketplace_fee_sum = 0.0
        process_initial_candidates = False  # Skip old processing
        
        # Ensure marketplace fees are counted even with path-only logic
        for flow in net_flows:
            if flow.is_marketplace_fee_account and flow.net_amount_ton > 0:
                marketplace_fee_sum += flow.net_amount_ton
        
        # DEBUG: Log all positive flows before filtering
        positive_flows = [f for f in net_flows if f.net_amount_ton > self.MIN_SIGNIFICANT_AMOUNT_TON]
        self.logger.debug(f"   🔍 Found {len(positive_flows)} positive flows before seller filtering")
        for flow in positive_flows:
            self.logger.debug(f"      {flow.address[-10:]}: {flow.net_amount_ton:.3f} TON, types={flow.flow_types}")
        
        if process_initial_candidates:
            for flow in initial_candidates:
                flow_addr_normalized = self._normalize_address(flow.address)
                
                # Fragment exception: Allow seller_credit flows even if they're in excluded_addresses
                has_seller_credit = any(ft in ['seller_credit', 'seller_credit_inferred'] for ft in flow.flow_types)
                is_excluded = flow_addr_normalized in excluded_addresses
                fragment_exception = has_seller_credit and any(h.marketplace_type == "fragment" for h in getattr(self, 'fee_hits', []))
                
                if (flow.net_amount_ton > self.MIN_SIGNIFICANT_AMOUNT_TON and 
                    (not is_excluded or fragment_exception)):
                    # Filter out nft_contract credits from seller candidates
                    has_nft_contract_only = flow.flow_types == ['nft_contract'] or 'nft_contract' in flow.flow_types
                    
                    if has_seller_credit or (not has_nft_contract_only and flow.flow_types):
                        seller_candidates.append(flow)
                        seller_note = " (Fragment exception)" if fragment_exception else ""
                        self.logger.debug(f"   ✅ Added seller candidate: {flow.address[-10:]} = {flow.net_amount_ton:.3f} TON{seller_note}")
                        if has_seller_credit:
                            # NFT CREDIT GUARD: Never include NFT contract addresses in seller credit sum
                            if 'nft_contract' not in flow.flow_types:
                                seller_credit_sum += flow.net_amount_ton
                            else:
                                self.logger.debug(f"   🚫 NFT CREDIT GUARD: Excluding NFT contract from seller sum: {flow.address[-10:]} = {flow.net_amount_ton:.3f} TON")
                elif flow.net_amount_ton > self.MIN_SIGNIFICANT_AMOUNT_TON:
                    exclusion_reason = "in excluded_addresses" if is_excluded else "below threshold"
                    self.logger.debug(f"   ❌ Excluded seller candidate: {flow.address[-10:]} = {flow.net_amount_ton:.3f} TON ({exclusion_reason})")
                
                # Track marketplace fees for fallback sale amount calculation
                if flow.is_marketplace_fee_account and flow.net_amount_ton > 0:
                    marketplace_fee_sum += flow.net_amount_ton
        
        # Handle marketplace fees for both paths (subtree and broadened)
        for flow in net_flows:
            if flow.is_marketplace_fee_account and flow.net_amount_ton > 0:
                marketplace_fee_sum += flow.net_amount_ton
        
        seller_flow = max(seller_candidates, key=lambda f: f.net_amount_ton, default=None)
        # Note: seller already determined by path-only logic above
        # seller = seller_flow.address if seller_flow else ""
        
        # NEW: Always use sum of all seller credits as sale amount
        if seller_credit_sum > 0:
            sale_amount = seller_credit_sum + marketplace_fee_sum
            self.logger.debug(f"   💰 Sale amount from ALL seller credits: seller_credits ({seller_credit_sum:.3f}) + fees ({marketplace_fee_sum:.3f}) = {sale_amount:.3f} TON")
        else:
            # Fallback to largest individual flow if no seller credits found
            sale_amount = seller_flow.net_amount_ton if seller_flow else 0.0
            self.logger.debug(f"   🔄 Fallback sale amount from largest flow: {sale_amount:.3f} TON")
            
        # If no seller found but we have seller credits, use largest seller credit address
        if not seller and seller_credit_sum > 0:
            seller_credit_flows = [f for f in net_flows if any(ft in ['seller_credit', 'seller_credit_inferred'] for ft in f.flow_types)]
            if seller_credit_flows:
                seller_flow = max(seller_credit_flows, key=lambda f: f.net_amount_ton)
                seller = seller_flow.address
                self.logger.debug(f"   🎯 Fallback seller from largest seller credit: {seller[-10:]}")
        
        # Find largest negative flow (buyer, excluding NFT, sale contract)
        buyer_candidates = []
        for flow in net_flows:
            flow_addr_normalized = self._normalize_address(flow.address)
            if (flow.net_amount_ton < -self.MIN_SIGNIFICANT_AMOUNT_TON and
                flow_addr_normalized not in excluded_addresses):
                buyer_candidates.append(flow)
        
        buyer_flow = min(buyer_candidates, key=lambda f: f.net_amount_ton, default=None)
        buyer = buyer_flow.address if buyer_flow else ""
        
        # NEW: Recalculate sale amount with proper subtree filtering and precision
        current_scope = getattr(self, '_current_analysis_scope', None)
        nft_address = current_scope.nft_address if current_scope else ''
        path_addr_set = {self._normalize_address(n.get("address", "")) for n in (current_scope.path_nodes or []) if n.get("address")} if current_scope else set()
        
        # Calculate sale amount as sum of valid seller credits in same subtree
        refined_sale_amount = sum(
            f.net_amount_ton for f in net_flows
            if (f.net_amount_ton > self.MIN_SIGNIFICANT_AMOUNT_TON
                and not f.is_marketplace_fee_account
                and self._normalize_address(f.address) not in {
                    self._normalize_address(nft_address or ''),
                    self._normalize_address(buyer or '')
                }
                and self._normalize_address(f.address) in path_addr_set)  # same subtree
        )
        
        if refined_sale_amount > 0:
            sale_amount = refined_sale_amount
            self.logger.debug(f"   🎯 REFINED sale amount (subtree-filtered): {sale_amount:.3f} TON")
        
        return seller, buyer, sale_amount
    
    def _calculate_fee_breakdown(self, marketplace_fees: List[NetFlow]) -> Dict[str, float]:
        """Calculate marketplace fees by type from net flows only"""
        fee_breakdown = {
            'fragment': 0.0,
            'marketapp': 0.0,
            'getgems': 0.0
        }
        
        for fee_flow in marketplace_fees:
            marketplace_type = fee_flow.marketplace_type
            if marketplace_type in fee_breakdown:
                fee_breakdown[marketplace_type] += fee_flow.net_amount_ton
        
        return fee_breakdown
    
    def _calculate_totals(self, net_flows: List[NetFlow]) -> Tuple[float, float]:
        """Calculate total positive and negative flows"""
        total_positive = sum(flow.net_amount_ton for flow in net_flows if flow.net_amount_ton > 0)
        total_negative = sum(flow.net_amount_ton for flow in net_flows if flow.net_amount_ton < 0)
        
        return total_positive, total_negative
    
    def _check_consistency(self, net_flows: List[NetFlow], marketplace_fees: List[NetFlow], sale_amount: float, buyer_payment: float = 0.0) -> Dict:
        """
        Perform consistency checks using explicit residual calculation
        
        Rule: Sum(included_flows) ≈ 0 (conservation of value)
        Excludes nft_contract flows to avoid internal hop double-counting
        """
        # NEW: Explicit residual calculation excluding nft_contract flows
        included_types = {'buyer_outgoing', 'seller_credit', 'seller_credit_inferred', 'marketplace_fee', 'gas_fee'}
        
        # Convert NetFlow to flow-like format for consistency with validation logic
        flows_for_check = []
        for nf in net_flows:
            # Determine flow_type based on NetFlow characteristics
            if nf.net_amount_ton < 0:
                flow_type = 'buyer_outgoing'
            elif 'nft_contract' in nf.flow_types:
                flow_type = 'nft_contract'  # Will be excluded
            elif any(ft in ['seller_credit', 'seller_credit_inferred'] for ft in nf.flow_types):
                flow_type = 'seller_credit'
            elif any(ft in ['marketplace_fee', 'gas_fee'] for ft in nf.flow_types):
                flow_type = 'marketplace_fee'
            else:
                flow_type = 'seller_credit'  # Default for positive flows
                
            flows_for_check.append({
                'flow_type': flow_type,
                'amount_nanoton': nf.net_amount_ton * 1e9
            })
        
        # Calculate residual using same logic as validation
        pos = sum(f['amount_nanoton'] for f in flows_for_check 
                 if f['flow_type'] in included_types and f['amount_nanoton'] > 0)
        neg = sum(f['amount_nanoton'] for f in flows_for_check 
                 if f['flow_type'] in included_types and f['amount_nanoton'] < 0)
        residual = pos + neg
        
        # Use buyer_payment for tolerance calculation
        buyer_payment_abs = max(abs(buyer_payment * 1e9), 1e9)  # Convert to nanoTON
        tolerance = max(1e8, 0.005 * buyer_payment_abs)  # 0.1 TON or 0.5%
        
        balance_error = residual / 1e9  # Convert back to TON for compatibility
        is_balanced = abs(residual) <= tolerance
        
        self.logger.debug(f"   🧮 Consistency check: residual={residual/1e9:.6f} TON (+{pos/1e9:.6f} / {neg/1e9:.6f}), tolerance={tolerance/1e9:.6f}, balanced={is_balanced}")
        
        flags = []
        is_consistent = True
        
        # Check 1: Value conservation using residual
        if not is_balanced:
            flags.append(f"Value conservation violated: {balance_error:.3f} TON imbalance")
            is_consistent = False
        
        # Check 2: Reasonable marketplace fees (should be < 50% of sale amount)
        total_fees = sum(fee.net_amount_ton for fee in marketplace_fees)
        if sale_amount > 0 and total_fees > 0.5 * sale_amount:
            flags.append(f"High marketplace fees: {total_fees:.3f} TON (>{50}% of sale)")
        
        # Check 3: Minimum flows for sales
        if sale_amount > 0 and len(net_flows) < 2:
            flags.append("Too few flows for a sale transaction")
        
        return {
            'is_consistent': is_consistent,
            'error_ton': abs(balance_error),
            'flags': flags
        }
    
    def convert_to_legacy_format(self, analysis: FinancialAnalysis) -> Dict:
        """Convert new analysis to legacy financial_info format for compatibility"""
        return {
            'buyer_address': analysis.buyer_address,
            'buyer_paid_ton': analysis.buyer_payment_ton,
            'fragment_received_ton': analysis.fragment_fee_ton,
            'marketapp_received_ton': analysis.marketapp_fee_ton,
            'getgems_received_ton': analysis.getgems_fee_ton,
            'nft_contract_received_ton': 0,  # Not implemented yet
            'total_amount_ton': analysis.buyer_payment_ton,
            'from_address': analysis.seller_address,
            'to_address': analysis.buyer_address,
            'transfer_type': 'direct'
        }
    
    


    def _extract_fragment_auction_fee(self, trace: Dict) -> Optional[float]:
        """
        Extract Fragment fee from auction_fill_up amount in mint_auction.
        
        In Fragment mint_auction:
        - Child node receives auction_fill_up message to Fragment address
        - The credit amount is the Fragment fee
        """
        def scan_node(node):
            if not isinstance(node, dict):
                return None
                
            tx = node.get('transaction', {})
            if not tx:
                return None
                
            # Check for auction_fill_up message to Fragment
            in_msg = tx.get('in_msg', {})
            if not in_msg:
                return None
                
            # Check if this is an auction_fill_up message
            decoded_op_name = in_msg.get('decoded_op_name', '')
            if decoded_op_name != 'auction_fill_up':
                return None
                
            # Check if destination is Fragment address
            destination = in_msg.get('destination', {})
            dest_address = destination.get('address', '')
            dest_name = destination.get('name', '')
            
            if dest_name != 'Fragment':
                return None
                
            # Extract credit amount (what Fragment receives)
            credit_phase = tx.get('credit_phase', {})
            credit_amount = credit_phase.get('credit', 0)
            
            try:
                credit_nanoton = int(credit_amount)
                credit_ton = credit_nanoton / 1e9
                if credit_ton > 0:
                    self.logger.debug(f"   💰 Fragment auction_fill_up: {credit_ton:.6f} TON to {dest_address[-10:]}")
                    return credit_ton
            except (ValueError, TypeError):
                pass
                
            return None
        
        # Traverse trace to find auction_fill_up to Fragment
        result = UnifiedTraversal.find_first(trace, scan_node)
        return result if result else None



    def _classify_marketplace_by_addr(self, address: str) -> str:
        """Classify marketplace type by fee address"""
        from .config import (
            FRAGMENT_FEE_ADDRESS, MARKETAPP_FEE_ADDRESS, MARKETAPP_MARKETPLACE_ADDRESS,
            GETGEMS_FEE_ADDRESS, GETGEMS_SALES_ADDRESS
        )
        
        normalized = self._normalize_address(address)
        
        if normalized == self._normalize_address(FRAGMENT_FEE_ADDRESS):
            return 'fragment'
        elif normalized in [self._normalize_address(MARKETAPP_FEE_ADDRESS), 
                           self._normalize_address(MARKETAPP_MARKETPLACE_ADDRESS)]:
            return 'marketapp'
        elif normalized in [self._normalize_address(GETGEMS_FEE_ADDRESS),
                           self._normalize_address(GETGEMS_SALES_ADDRESS)]:
            return 'getgems'
        else:
            return 'unknown'
    
    # REMOVED: classify_marketplace_type moved to ImprovedMarketplaceClassifier
    # Use: classifier.classify_from_financial(analysis, comments) instead