"""
Error Classification and Analysis Module

This module provides comprehensive error handling capabilities:
- Error categorization by type and severity
- Root cause analysis
- Error pattern detection
- Suggested fixes based on error type

Author: Data Science Agent System
Created: 2025-10-19
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field
from loguru import logger
import re
import traceback


class ErrorCategory(str, Enum):
    """Categories of errors that can occur during execution"""
    
    # Code Execution Errors
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    NAME_ERROR = "name_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    ATTRIBUTE_ERROR = "attribute_error"
    KEY_ERROR = "key_error"
    INDEX_ERROR = "index_error"
    
    # Resource Errors
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_ERROR = "permission_error"
    
    # Data Errors
    DATA_VALIDATION = "data_validation"
    DATA_FORMAT = "data_format"
    MISSING_DATA = "missing_data"
    DATA_TYPE_MISMATCH = "data_type_mismatch"
    
    # API/Network Errors
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    RATE_LIMIT = "rate_limit"
    
    # Workflow Errors
    DEPENDENCY_FAILED = "dependency_failed"
    INVALID_STATE = "invalid_state"
    CONFIGURATION_ERROR = "configuration_error"
    
    # Unknown
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Severity levels for errors"""
    
    CRITICAL = "critical"  # Cannot continue, requires manual intervention
    HIGH = "high"          # Major issue, may retry with modifications
    MEDIUM = "medium"      # Recoverable issue, can retry
    LOW = "low"            # Minor issue, easily recoverable


class RecoveryStrategy(str, Enum):
    """Strategies for recovering from errors"""
    
    RETRY_SAME = "retry_same"                    # Retry with same parameters
    RETRY_MODIFIED = "retry_modified"            # Retry with modified parameters
    SKIP_AND_CONTINUE = "skip_and_continue"      # Skip this step
    USE_FALLBACK = "use_fallback"                # Use alternative approach
    REQUEST_INPUT = "request_input"              # Ask for user input
    FAIL = "fail"                                # Cannot recover


class ErrorAnalysis(BaseModel):
    """Comprehensive error analysis result"""
    
    # Error identification
    category: ErrorCategory
    severity: ErrorSeverity
    error_message: str
    stack_trace: Optional[str] = None
    
    # Analysis
    root_cause: str
    probable_reasons: List[str] = Field(default_factory=list)
    affected_components: List[str] = Field(default_factory=list)
    
    # Recovery
    recovery_strategy: RecoveryStrategy
    suggested_fixes: List[str] = Field(default_factory=list)
    code_modifications: Optional[str] = None
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    context: Dict[str, Any] = Field(default_factory=dict)
    is_recoverable: bool = True


class ErrorAnalyzer:
    """
    Analyzes errors and provides detailed diagnostics and recovery suggestions.
    
    Features:
    - Categorizes errors by type
    - Analyzes root causes
    - Suggests recovery strategies
    - Provides code fixes where applicable
    """
    
    def __init__(self):
        """Initialize error analyzer"""
        self.error_patterns = self._build_error_patterns()
        self.fix_templates = self._build_fix_templates()
        logger.info("ErrorAnalyzer initialized")
    
    def analyze_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None
    ) -> ErrorAnalysis:
        """
        Analyze an error and provide comprehensive diagnostics.
        
        Args:
            error: The exception that occurred
            context: Additional context (stage, inputs, etc.)
            code: The code that caused the error (if applicable)
            
        Returns:
            ErrorAnalysis with detailed diagnostics and recovery suggestions
        """
        error_str = str(error)
        error_type = type(error).__name__
        stack_trace = traceback.format_exc()
        
        logger.info(f"Analyzing error: {error_type}: {error_str}")
        
        # Categorize error
        category = self._categorize_error(error_type, error_str)
        
        # Determine severity
        severity = self._determine_severity(category, error_str, context)
        
        # Analyze root cause
        root_cause = self._analyze_root_cause(error_type, error_str, code, stack_trace)
        
        # Get probable reasons
        probable_reasons = self._get_probable_reasons(category, error_str, code)
        
        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(category, severity)
        
        # Get suggested fixes
        suggested_fixes = self._generate_fixes(category, error_str, code)
        
        # Generate code modifications if possible
        code_modifications = self._generate_code_fix(
            category, error_str, code
        ) if code else None
        
        # Determine if recoverable
        is_recoverable = severity != ErrorSeverity.CRITICAL and \
                        recovery_strategy != RecoveryStrategy.FAIL
        
        analysis = ErrorAnalysis(
            category=category,
            severity=severity,
            error_message=error_str,
            stack_trace=stack_trace,
            root_cause=root_cause,
            probable_reasons=probable_reasons,
            affected_components=self._identify_affected_components(stack_trace),
            recovery_strategy=recovery_strategy,
            suggested_fixes=suggested_fixes,
            code_modifications=code_modifications,
            timestamp=datetime.now(),
            context=context or {},
            is_recoverable=is_recoverable
        )
        
        logger.info(f"Error analysis complete: {category.value}, "
                   f"severity={severity.value}, recoverable={is_recoverable}")
        
        return analysis
    
    def _categorize_error(self, error_type: str, error_msg: str) -> ErrorCategory:
        """Categorize error based on type and message"""
        
        # Direct type mapping
        type_mapping = {
            "SyntaxError": ErrorCategory.SYNTAX_ERROR,
            "ImportError": ErrorCategory.IMPORT_ERROR,
            "ModuleNotFoundError": ErrorCategory.IMPORT_ERROR,
            "NameError": ErrorCategory.NAME_ERROR,
            "TypeError": ErrorCategory.TYPE_ERROR,
            "ValueError": ErrorCategory.VALUE_ERROR,
            "AttributeError": ErrorCategory.ATTRIBUTE_ERROR,
            "KeyError": ErrorCategory.KEY_ERROR,
            "IndexError": ErrorCategory.INDEX_ERROR,
            "FileNotFoundError": ErrorCategory.FILE_NOT_FOUND,
            "PermissionError": ErrorCategory.PERMISSION_ERROR,
            "MemoryError": ErrorCategory.MEMORY_ERROR,
            "TimeoutError": ErrorCategory.TIMEOUT,
        }
        
        if error_type in type_mapping:
            return type_mapping[error_type]
        
        # Pattern matching on error message
        for pattern, category in self.error_patterns.items():
            if re.search(pattern, error_msg, re.IGNORECASE):
                return category
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(
        self,
        category: ErrorCategory,
        error_msg: str,
        context: Optional[Dict[str, Any]]
    ) -> ErrorSeverity:
        """Determine error severity"""
        
        # Critical errors (cannot recover)
        critical_categories = {
            ErrorCategory.MEMORY_ERROR,
            ErrorCategory.CONFIGURATION_ERROR,
        }
        
        if category in critical_categories:
            return ErrorSeverity.CRITICAL
        
        # High severity
        high_severity_categories = {
            ErrorCategory.FILE_NOT_FOUND,
            ErrorCategory.PERMISSION_ERROR,
            ErrorCategory.API_ERROR,
            ErrorCategory.DEPENDENCY_FAILED,
        }
        
        if category in high_severity_categories:
            return ErrorSeverity.HIGH
        
        # Medium severity (recoverable with retry)
        medium_severity_categories = {
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK_ERROR,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.TYPE_ERROR,
            ErrorCategory.VALUE_ERROR,
        }
        
        if category in medium_severity_categories:
            return ErrorSeverity.MEDIUM
        
        # Low severity (easily fixable)
        return ErrorSeverity.LOW
    
    def _analyze_root_cause(
        self,
        error_type: str,
        error_msg: str,
        code: Optional[str],
        stack_trace: str
    ) -> str:
        """Analyze the root cause of the error"""
        
        # Extract line number from stack trace if available
        line_match = re.search(r'line (\d+)', stack_trace)
        line_num = line_match.group(1) if line_match else "unknown"
        
        # Build root cause description
        root_cause = f"{error_type} at line {line_num}: {error_msg}"
        
        # Add code context if available
        if code and line_num != "unknown":
            try:
                lines = code.split('\n')
                line_idx = int(line_num) - 1
                if 0 <= line_idx < len(lines):
                    problematic_line = lines[line_idx].strip()
                    root_cause += f"\n  Problematic code: {problematic_line}"
            except:
                pass
        
        return root_cause
    
    def _get_probable_reasons(
        self,
        category: ErrorCategory,
        error_msg: str,
        code: Optional[str]
    ) -> List[str]:
        """Get probable reasons for the error"""
        
        reasons_map = {
            ErrorCategory.IMPORT_ERROR: [
                "Required package not installed",
                "Incorrect package name or version",
                "Package not in Python path",
            ],
            ErrorCategory.NAME_ERROR: [
                "Variable not defined before use",
                "Typo in variable name",
                "Variable out of scope",
            ],
            ErrorCategory.TYPE_ERROR: [
                "Incorrect data type passed to function",
                "Operation not supported for data type",
                "Missing required type conversion",
            ],
            ErrorCategory.FILE_NOT_FOUND: [
                "File path is incorrect",
                "File has not been created yet",
                "Working directory is wrong",
                "File was deleted or moved",
            ],
            ErrorCategory.TIMEOUT: [
                "Operation taking longer than expected",
                "Infinite loop in code",
                "Large dataset processing",
                "Network latency",
            ],
            ErrorCategory.KEY_ERROR: [
                "Dictionary key doesn't exist",
                "Column name not in DataFrame",
                "Incorrect key name (case sensitivity)",
            ],
            ErrorCategory.INDEX_ERROR: [
                "List/array index out of bounds",
                "Empty list/array accessed",
                "Off-by-one error in indexing",
            ],
        }
        
        return reasons_map.get(category, ["Unknown cause"])
    
    def _determine_recovery_strategy(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity
    ) -> RecoveryStrategy:
        """Determine the best recovery strategy"""
        
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.FAIL
        
        # Strategy mapping by category
        strategy_map = {
            ErrorCategory.TIMEOUT: RecoveryStrategy.RETRY_MODIFIED,
            ErrorCategory.NETWORK_ERROR: RecoveryStrategy.RETRY_SAME,
            ErrorCategory.RATE_LIMIT: RecoveryStrategy.RETRY_MODIFIED,
            ErrorCategory.IMPORT_ERROR: RecoveryStrategy.USE_FALLBACK,
            ErrorCategory.FILE_NOT_FOUND: RecoveryStrategy.REQUEST_INPUT,
            ErrorCategory.TYPE_ERROR: RecoveryStrategy.RETRY_MODIFIED,
            ErrorCategory.VALUE_ERROR: RecoveryStrategy.RETRY_MODIFIED,
            ErrorCategory.SYNTAX_ERROR: RecoveryStrategy.RETRY_MODIFIED,
            ErrorCategory.NAME_ERROR: RecoveryStrategy.RETRY_MODIFIED,
        }
        
        return strategy_map.get(category, RecoveryStrategy.RETRY_SAME)
    
    def _generate_fixes(
        self,
        category: ErrorCategory,
        error_msg: str,
        code: Optional[str]
    ) -> List[str]:
        """Generate suggested fixes for the error"""
        
        fixes = []
        
        # Get template fixes for this category
        template_fixes = self.fix_templates.get(category, [])
        fixes.extend(template_fixes)
        
        # Add specific fixes based on error message
        if "not found" in error_msg.lower():
            fixes.append("Verify the file/variable/package name is correct")
        
        if "timeout" in error_msg.lower():
            fixes.append("Increase timeout duration")
            fixes.append("Optimize code for better performance")
            fixes.append("Process data in smaller chunks")
        
        if "memory" in error_msg.lower():
            fixes.append("Reduce dataset size")
            fixes.append("Use chunked processing")
            fixes.append("Clear unnecessary variables")
        
        return fixes[:5]  # Limit to top 5 fixes
    
    def _generate_code_fix(
        self,
        category: ErrorCategory,
        error_msg: str,
        code: str
    ) -> Optional[str]:
        """Generate modified code to fix the error (if possible)"""
        
        # This is a simplified version - in production, you'd use
        # more sophisticated code analysis and generation
        
        if category == ErrorCategory.IMPORT_ERROR:
            # Extract package name from error
            match = re.search(r"No module named '(.+)'", error_msg)
            if match:
                package = match.group(1)
                return f"# Install missing package: pip install {package}\n{code}"
        
        if category == ErrorCategory.FILE_NOT_FOUND:
            # Add path validation
            return f"""import os
# Validate file exists before processing
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {{file_path}}")

{code}"""
        
        
        return None
    
    def _identify_affected_components(self, stack_trace: str) -> List[str]:
        """Identify which components/modules are affected"""
        
        components = []
        
        # Extract module names from stack trace
        module_pattern = r'File ".*/([\w_]+)\.py"'
        matches = re.findall(module_pattern, stack_trace)
        
        components = list(set(matches))  # Remove duplicates
        
        return components
    
    def _build_error_patterns(self) -> Dict[str, ErrorCategory]:
        """Build regex patterns for error categorization"""
        
        return {
            r"timeout|timed out": ErrorCategory.TIMEOUT,
            r"connection refused|network": ErrorCategory.NETWORK_ERROR,
            r"rate limit|too many requests": ErrorCategory.RATE_LIMIT,
            r"no such file|file not found": ErrorCategory.FILE_NOT_FOUND,
            r"permission denied": ErrorCategory.PERMISSION_ERROR,
            r"out of memory|memory error": ErrorCategory.MEMORY_ERROR,
            r"data validation|invalid data": ErrorCategory.DATA_VALIDATION,
            r"missing column|column not found": ErrorCategory.MISSING_DATA,
            r"dependency failed": ErrorCategory.DEPENDENCY_FAILED,
        }
    
    def _build_fix_templates(self) -> Dict[ErrorCategory, List[str]]:
        """Build template fixes for each error category"""
        
        return {
            ErrorCategory.IMPORT_ERROR: [
                "Install the missing package: pip install <package>",
                "Check package name spelling",
                "Verify package is compatible with Python version",
            ],
            ErrorCategory.TIMEOUT: [
                "Increase timeout duration",
                "Optimize code performance",
                "Process data in smaller batches",
            ],
            ErrorCategory.FILE_NOT_FOUND: [
                "Verify file path is correct",
                "Check file exists before accessing",
                "Use absolute paths instead of relative",
            ],
            ErrorCategory.NAME_ERROR: [
                "Define variable before using it",
                "Check for typos in variable name",
                "Verify variable is in correct scope",
            ],
            ErrorCategory.TYPE_ERROR: [
                "Add type conversion: int(), str(), float()",
                "Validate input types before processing",
                "Check function signature matches arguments",
            ],
            ErrorCategory.KEY_ERROR: [
                "Verify key exists: if key in dict",
                "Use dict.get(key, default) for safe access",
                "Check for case sensitivity in keys",
            ],
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    analyzer = ErrorAnalyzer()
    
    # Simulate an error
    try:
        import nonexistent_package
    except Exception as e:
        analysis = analyzer.analyze_error(e)
        
        print(f"Category: {analysis.category}")
        print(f"Severity: {analysis.severity}")
        print(f"Root cause: {analysis.root_cause}")
        print(f"Recovery: {analysis.recovery_strategy}")
        print(f"Fixes: {analysis.suggested_fixes}")