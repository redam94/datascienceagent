"""
Retry Strategies Module

This module provides flexible retry mechanisms for handling transient failures:
- Configurable retry policies
- Exponential backoff with jitter
- Adaptive retry based on error type
- Retry budget management
- Circuit breaker pattern

Author: Data Science Agent System
Created: 2025-10-19
"""

import asyncio
import time
import random
from enum import Enum
from typing import Optional, Callable, Any, Dict, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from loguru import logger

from datascienceagent.core.error_handling import ErrorCategory, ErrorSeverity, ErrorAnalyzer


class RetryPolicy(str, Enum):
    """Retry policy types"""
    
    FIXED = "fixed"                    # Fixed delay between retries
    EXPONENTIAL = "exponential"        # Exponential backoff
    LINEAR = "linear"                  # Linear increase in delay
    ADAPTIVE = "adaptive"              # Adapt based on error type


class CircuitState(str, Enum):
    """Circuit breaker states"""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, stop trying
    HALF_OPEN = "half_open"  # Testing if recovered


class RetryConfig(BaseModel):
    """Configuration for retry behavior"""
    
    # Basic retry settings
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    
    # Policy
    policy: RetryPolicy = RetryPolicy.EXPONENTIAL
    
    # Exponential backoff settings
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    
    # Adaptive settings
    adapt_to_error_type: bool = True
    
    # Circuit breaker settings
    enable_circuit_breaker: bool = False
    failure_threshold: int = 5  # Open circuit after N failures
    success_threshold: int = 2  # Close circuit after N successes in half-open
    timeout_seconds: float = 60.0  # Time to wait before trying half-open
    
    # Budget
    max_total_retry_time: Optional[float] = 300.0  # Max 5 minutes of retrying
    
    # Logging
    log_retries: bool = True


class RetryAttempt(BaseModel):
    """Record of a retry attempt"""
    
    attempt_number: int
    timestamp: datetime = Field(default_factory=datetime.now)
    delay_seconds: float
    error_category: Optional[ErrorCategory] = None
    error_message: str = ""
    success: bool = False


class RetryResult(BaseModel):
    """Result of retry operation"""
    
    success: bool
    result: Any = None
    total_attempts: int
    total_time_seconds: float
    attempts: List[RetryAttempt] = Field(default_factory=list)
    final_error: Optional[str] = None
    gave_up_reason: Optional[str] = None


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Too many failures, reject immediately
    - HALF_OPEN: Testing if system recovered
    """
    
    def __init__(self, config: RetryConfig):
        """Initialize circuit breaker"""
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        logger.info("CircuitBreaker initialized")
    
    def can_proceed(self) -> bool:
        """Check if operation can proceed"""
        
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    return True
            return False
        
        # HALF_OPEN state
        return True
    
    def record_success(self):
        """Record successful operation"""
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(f"Circuit breaker: success {self.success_count}/"
                       f"{self.config.success_threshold}")
            
            if self.success_count >= self.config.success_threshold:
                logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation"""
        
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker: HALF_OPEN -> OPEN (failure during test)")
            self.state = CircuitState.OPEN
            self.success_count = 0
        
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            logger.info(f"Circuit breaker: failure {self.failure_count}/"
                       f"{self.config.failure_threshold}")
            
            if self.failure_count >= self.config.failure_threshold:
                logger.warning("Circuit breaker: CLOSED -> OPEN (too many failures)")
                self.state = CircuitState.OPEN


class RetryManager:
    """
    Manages retry logic with configurable strategies and error analysis.
    
    Features:
    - Multiple retry policies (fixed, exponential, linear, adaptive)
    - Exponential backoff with jitter
    - Circuit breaker pattern
    - Error-aware retry decisions
    - Retry budget management
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry manager.
        
        Args:
            config: Retry configuration (uses defaults if None)
        """
        self.config = config or RetryConfig()
        self.error_analyzer = ErrorAnalyzer()
        
        # Circuit breaker
        self.circuit_breaker: Optional[CircuitBreaker] = None
        if self.config.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(self.config)
        
        logger.info(f"RetryManager initialized with policy={self.config.policy.value}, "
                   f"max_attempts={self.config.max_attempts}")
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        error_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetryResult:
        """
        Execute a function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            error_context: Context for error analysis
            **kwargs: Keyword arguments for func
            
        Returns:
            RetryResult with execution outcome and retry history
        """
        start_time = time.time()
        attempts: List[RetryAttempt] = []
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            # Check circuit breaker
            if self.circuit_breaker and not self.circuit_breaker.can_proceed():
                logger.warning("Circuit breaker is OPEN, aborting retry")
                return RetryResult(
                    success=False,
                    total_attempts=attempt_num - 1,
                    total_time_seconds=time.time() - start_time,
                    attempts=attempts,
                    gave_up_reason="Circuit breaker is OPEN"
                )
            
            # Check retry budget
            elapsed = time.time() - start_time
            if (self.config.max_total_retry_time and 
                elapsed >= self.config.max_total_retry_time):
                logger.warning(f"Retry budget exhausted ({elapsed:.1f}s)")
                return RetryResult(
                    success=False,
                    total_attempts=attempt_num - 1,
                    total_time_seconds=elapsed,
                    attempts=attempts,
                    gave_up_reason="Retry time budget exceeded"
                )
            
            try:
                if self.config.log_retries:
                    logger.info(f"Attempt {attempt_num}/{self.config.max_attempts}")
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Success!
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                attempts.append(RetryAttempt(
                    attempt_number=attempt_num,
                    delay_seconds=0,
                    success=True
                ))
                
                logger.info(f"✅ Success on attempt {attempt_num}")
                
                return RetryResult(
                    success=True,
                    result=result,
                    total_attempts=attempt_num,
                    total_time_seconds=time.time() - start_time,
                    attempts=attempts
                )
            
            except Exception as e:
                # Analyze error
                analysis = self.error_analyzer.analyze_error(
                    e,
                    context=error_context
                )
                
                logger.error(f"❌ Attempt {attempt_num} failed: {analysis.error_message}")
                
                # Record failure in circuit breaker
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                
                # Check if we should retry
                should_retry = self._should_retry(
                    analysis,
                    attempt_num,
                    self.config.max_attempts
                )
                
                if not should_retry:
                    logger.warning(f"Not retrying due to: {analysis.category.value}")
                    attempts.append(RetryAttempt(
                        attempt_number=attempt_num,
                        delay_seconds=0,
                        error_category=analysis.category,
                        error_message=str(e),
                        success=False
                    ))
                    
                    return RetryResult(
                        success=False,
                        total_attempts=attempt_num,
                        total_time_seconds=time.time() - start_time,
                        attempts=attempts,
                        final_error=str(e),
                        gave_up_reason=f"Error not retryable: {analysis.category.value}"
                    )
                
                # Calculate delay before next retry
                if attempt_num < self.config.max_attempts:
                    delay = self._calculate_delay(
                        attempt_num,
                        analysis.category
                    )
                    
                    attempts.append(RetryAttempt(
                        attempt_number=attempt_num,
                        delay_seconds=delay,
                        error_category=analysis.category,
                        error_message=str(e),
                        success=False
                    ))
                    
                    if self.config.log_retries:
                        logger.info(f"Retrying in {delay:.1f}s...")
                    
                    await asyncio.sleep(delay)
                else:
                    # Last attempt failed
                    attempts.append(RetryAttempt(
                        attempt_number=attempt_num,
                        delay_seconds=0,
                        error_category=analysis.category,
                        error_message=str(e),
                        success=False
                    ))
        
        # All attempts exhausted
        return RetryResult(
            success=False,
            total_attempts=self.config.max_attempts,
            total_time_seconds=time.time() - start_time,
            attempts=attempts,
            final_error=str(e),
            gave_up_reason=f"Max attempts ({self.config.max_attempts}) reached"
        )
    
    def _should_retry(
        self,
        analysis: Any,  # ErrorAnalysis
        attempt_num: int,
        max_attempts: int
    ) -> bool:
        """Determine if we should retry based on error analysis"""
        
        # Don't retry critical errors
        if analysis.severity == ErrorSeverity.CRITICAL:
            return False
        
        # Don't retry if not recoverable
        if not analysis.is_recoverable:
            return False
        
        # Don't retry certain error categories
        no_retry_categories = {
            ErrorCategory.SYNTAX_ERROR,
            ErrorCategory.CONFIGURATION_ERROR,
            ErrorCategory.PERMISSION_ERROR,
        }
        
        if analysis.category in no_retry_categories:
            return False
        
        # Always retry if we have attempts left and error is retryable
        return attempt_num < max_attempts
    
    def _calculate_delay(
        self,
        attempt_num: int,
        error_category: ErrorCategory
    ) -> float:
        """Calculate delay before next retry"""
        
        delay = 0.0
        
        if self.config.policy == RetryPolicy.FIXED:
            delay = self.config.base_delay
        
        elif self.config.policy == RetryPolicy.LINEAR:
            delay = self.config.base_delay * attempt_num
        
        elif self.config.policy == RetryPolicy.EXPONENTIAL:
            delay = self.config.base_delay * (
                self.config.exponential_base ** (attempt_num - 1)
            )
        
        elif self.config.policy == RetryPolicy.ADAPTIVE:
            # Adapt delay based on error type
            delay = self._adaptive_delay(attempt_num, error_category)
        
        # Apply maximum delay cap
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)  # Ensure non-negative
    
    def _adaptive_delay(
        self,
        attempt_num: int,
        error_category: ErrorCategory
    ) -> float:
        """Calculate adaptive delay based on error type"""
        
        # Base exponential delay
        base_delay = self.config.base_delay * (
            self.config.exponential_base ** (attempt_num - 1)
        )
        
        # Adjust based on error category
        if error_category == ErrorCategory.RATE_LIMIT:
            # Longer delay for rate limits
            return base_delay * 2.0
        
        elif error_category == ErrorCategory.NETWORK_ERROR:
            # Medium delay for network issues
            return base_delay * 1.5
        
        elif error_category == ErrorCategory.TIMEOUT:
            # Longer delay for timeouts
            return base_delay * 1.8
        
        elif error_category in {ErrorCategory.TYPE_ERROR, ErrorCategory.VALUE_ERROR}:
            # Shorter delay for logic errors (likely need code fix)
            return base_delay * 0.5
        
        return base_delay


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_standard_retry_config() -> RetryConfig:
    """Create standard retry configuration for most use cases"""
    return RetryConfig(
        max_attempts=6,
        base_delay=1.0,
        max_delay=30.0,
        policy=RetryPolicy.EXPONENTIAL,
        exponential_base=2.0,
        jitter=True,
        adapt_to_error_type=True,
        log_retries=True
    )


def create_aggressive_retry_config() -> RetryConfig:
    """Create aggressive retry configuration for critical operations"""
    return RetryConfig(
        max_attempts=5,
        base_delay=0.5,
        max_delay=60.0,
        policy=RetryPolicy.EXPONENTIAL,
        exponential_base=2.0,
        jitter=True,
        adapt_to_error_type=True,
        enable_circuit_breaker=True,
        failure_threshold=10,
        max_total_retry_time=600.0,  # 10 minutes
        log_retries=True
    )


def create_conservative_retry_config() -> RetryConfig:
    """Create conservative retry configuration for non-critical operations"""
    return RetryConfig(
        max_attempts=2,
        base_delay=2.0,
        max_delay=30.0,
        policy=RetryPolicy.FIXED,
        jitter=False,
        adapt_to_error_type=False,
        max_total_retry_time=120.0,  # 2 minutes
        log_retries=True
    )


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    async def example_usage():
        # Create retry manager
        retry_manager = RetryManager(create_standard_retry_config())
        
        # Example function that might fail
        async def unreliable_operation():
            if random.random() < 0.5:  # 70% failure rate
                raise Exception("Random failure!")
            return "Success!"
        
        # Execute with retry
        result = await retry_manager.execute_with_retry(unreliable_operation)
        
        print(f"Success: {result.success}")
        print(f"Attempts: {result.total_attempts}")
        print(f"Time: {result.total_time_seconds:.2f}s")
        if result.success:
            print(f"Result: {result.result}")
        else:
            print(f"Failed: {result.gave_up_reason}")
    
    asyncio.run(example_usage())