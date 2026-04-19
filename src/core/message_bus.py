"""
ROS-Style Message Bus — Thread-safe Publish/Subscribe System

Mimics ROS topic communication for Windows development.
Each topic supports multiple subscribers and maintains message queues.
Drop-in replaceable with actual rospy when deploying on ROS.
"""

import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional
import queue
import logging

logger = logging.getLogger(__name__)


class MessageBus:
    """
    Thread-safe publish/subscribe message bus mimicking ROS topics.
    
    Supports:
    - Topic-based pub/sub with callback registration
    - Synchronous and asynchronous message delivery
    - Message history for late subscribers (latch mode)
    - Thread-safe operations for parallel node execution
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern — one message bus per system."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # {topic_name: [callback_functions]}
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        # {topic_name: last_message} for latched topics
        self._latched: Dict[str, Any] = {}
        # Latched topic set
        self._latch_topics: set = set()
        # Lock per topic for fine-grained concurrency
        self._topic_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        # Global lock for subscriber registration
        self._sub_lock = threading.Lock()
        # Message counters for diagnostics
        self._msg_counts: Dict[str, int] = defaultdict(int)
        # Timing data
        self._pub_times: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("MessageBus initialized (singleton)")
    
    def publish(self, topic: str, message: Any) -> None:
        """
        Publish a message to a topic.
        All registered callbacks are invoked synchronously in the publisher's thread.
        
        Args:
            topic: Topic name (e.g., '/camera/image', '/detection/weeds')
            message: Any Python object to send
        """
        start = time.perf_counter()
        
        with self._topic_locks[topic]:
            self._msg_counts[topic] += 1
            
            if topic in self._latch_topics:
                self._latched[topic] = message
            
            with self._sub_lock:
                callbacks = list(self._subscribers.get(topic, []))
        
        # Invoke callbacks outside the lock to prevent deadlocks
        for callback in callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Callback error on topic '{topic}': {e}")
        
        elapsed = (time.perf_counter() - start) * 1000
        self._pub_times[topic].append(elapsed)
    
    def subscribe(self, topic: str, callback: Callable) -> None:
        """
        Subscribe to a topic with a callback function.
        
        Args:
            topic: Topic name to subscribe to
            callback: Function to invoke when a message is published
        """
        with self._sub_lock:
            self._subscribers[topic].append(callback)
            logger.debug(f"Subscribed to '{topic}' ({len(self._subscribers[topic])} subscribers)")
        
        # Deliver latched message if available
        if topic in self._latched:
            try:
                callback(self._latched[topic])
            except Exception as e:
                logger.error(f"Latched delivery error on '{topic}': {e}")
    
    def set_latch(self, topic: str) -> None:
        """Enable latching for a topic — new subscribers get the last message."""
        self._latch_topics.add(topic)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics for diagnostics."""
        stats = {}
        for topic in self._msg_counts:
            times = self._pub_times.get(topic, [])
            stats[topic] = {
                'message_count': self._msg_counts[topic],
                'avg_latency_ms': sum(times) / len(times) if times else 0,
                'max_latency_ms': max(times) if times else 0,
                'subscriber_count': len(self._subscribers.get(topic, []))
            }
        return stats
    
    def reset(self) -> None:
        """Reset the message bus (for testing)."""
        with self._sub_lock:
            self._subscribers.clear()
            self._latched.clear()
            self._latch_topics.clear()
            self._msg_counts.clear()
            self._pub_times.clear()
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
