from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
from typing import Callable
import logging

# Metrics
DOCUMENT_PROCESSING_TIME = Histogram(
    'document_processing_seconds',
    'Time spent processing documents',
    ['mime_type']
)

DOCUMENT_PROCESSING_ERRORS = Counter(
    'document_processing_errors_total',
    'Total number of document processing errors',
    ['mime_type', 'error_type']
)

ACTIVE_BATCH_JOBS = Gauge(
    'active_batch_jobs',
    'Number of active batch processing jobs'
)

SEARCH_REQUESTS = Counter(
    'search_requests_total',
    'Total number of search requests',
    ['status']
)

def track_time(metric: Histogram) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                metric.observe(time.time() - start_time)
                return result
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

class MetricsMiddleware:
    async def __call__(self, request, call_next):
        try:
            response = await call_next(request)
            SEARCH_REQUESTS.labels(
                status=response.status_code
            ).inc()
            return response
        except Exception as e:
            SEARCH_REQUESTS.labels(
                status=500
            ).inc()
            raise 