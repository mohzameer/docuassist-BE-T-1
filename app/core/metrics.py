from prometheus_client import Counter, Histogram, Info
import time

# Define metrics
REQUEST_COUNT = Counter(
    'app_request_count',
    'Application Request Count',
    ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Application Request Latency',
    ['method', 'endpoint']
)

def track_request_metrics():
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                response = await func(*args, **kwargs)
                REQUEST_COUNT.labels(
                    method=kwargs.get('method', 'unknown'),
                    endpoint=kwargs.get('endpoint', 'unknown'),
                    http_status=response.status_code
                ).inc()
                return response
            finally:
                REQUEST_LATENCY.labels(
                    method=kwargs.get('method', 'unknown'),
                    endpoint=kwargs.get('endpoint', 'unknown')
                ).observe(time.time() - start_time)
        return wrapper
    return decorator 