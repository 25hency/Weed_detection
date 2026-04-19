"""
ROS-Style Node Base Class

All modules inherit from this base class, which provides:
- Message bus integration (pub/sub)
- Threaded execution loop
- Lifecycle management (start/stop)
- Performance timing
"""

import threading
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from .message_bus import MessageBus

logger = logging.getLogger(__name__)


class NodeBase(ABC):
    """
    Abstract base class for all system nodes.
    Mirrors rospy.Node lifecycle with init, spin, and shutdown.
    """
    
    def __init__(self, node_name: str, rate_hz: float = 10.0):
        """
        Initialize a node.
        
        Args:
            node_name: Unique name for this node
            rate_hz: Target execution rate in Hz (for spin loop)
        """
        self.node_name = node_name
        self.rate_hz = rate_hz
        self.bus = MessageBus()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cycle_times: list = []
        self._start_time: float = 0
        
        logger.info(f"Node '{node_name}' created (rate={rate_hz}Hz)")
    
    def publish(self, topic: str, message: Any) -> None:
        """Publish a message to the bus."""
        self.bus.publish(topic, message)
    
    def subscribe(self, topic: str, callback) -> None:
        """Subscribe to a topic on the bus."""
        self.bus.subscribe(topic, callback)
    
    @abstractmethod
    def on_start(self) -> None:
        """Called once when the node starts. Override for initialization."""
        pass
    
    @abstractmethod
    def on_update(self, dt: float) -> None:
        """
        Called every cycle. Override with node logic.
        
        Args:
            dt: Time since last update in seconds
        """
        pass
    
    def on_shutdown(self) -> None:
        """Called when node is stopping. Override for cleanup."""
        pass
    
    def start(self) -> None:
        """Start the node in a background thread."""
        if self._running:
            logger.warning(f"Node '{self.node_name}' already running")
            return
        
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._spin_loop,
            name=f"Node-{self.node_name}",
            daemon=True
        )
        self._thread.start()
        logger.info(f"Node '{self.node_name}' started")
    
    def stop(self) -> None:
        """Stop the node and wait for thread to finish."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.on_shutdown()
        logger.info(f"Node '{self.node_name}' stopped")
    
    def _spin_loop(self) -> None:
        """Main execution loop — runs on_update at the target rate."""
        self.on_start()
        period = 1.0 / self.rate_hz
        last_time = time.perf_counter()
        
        while self._running:
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            
            cycle_start = time.perf_counter()
            try:
                self.on_update(dt)
            except Exception as e:
                logger.error(f"Node '{self.node_name}' update error: {e}")
            
            cycle_time = (time.perf_counter() - cycle_start) * 1000  # ms
            self._cycle_times.append(cycle_time)
            
            # Rate limiting
            elapsed = time.perf_counter() - now
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def uptime(self) -> float:
        """Node uptime in seconds."""
        if self._start_time == 0:
            return 0
        return time.time() - self._start_time
    
    def get_timing_stats(self) -> dict:
        """Get performance statistics for this node."""
        if not self._cycle_times:
            return {'avg_ms': 0, 'max_ms': 0, 'min_ms': 0, 'count': 0}
        
        recent = self._cycle_times[-100:]  # Last 100 cycles
        return {
            'avg_ms': sum(recent) / len(recent),
            'max_ms': max(recent),
            'min_ms': min(recent),
            'count': len(self._cycle_times)
        }
