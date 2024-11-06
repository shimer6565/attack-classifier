from gemini import generate_context
from summarize_text import summarize_alert
from RAG_fusion import alert_retriever


def alert_pipeline(alert_json):
    alert_context = summarize_alert(alert_json)
    print("HIiI", alert_context)
    similiar_alerts = alert_retriever(alert_context)
    print(similiar_alerts)
    result = generate_context(similiar_alerts, alert_context)
    print(result)


print(
    alert_pipeline(
        """
    (
        {

        "event_time": "2024-10-25T12:34:56.789Z",
        "event_type": "http_request",
        "source_ip": "192.168.1.100",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        "http_method": "GET",
        "url": "/api/data",
        "http_version": "HTTP/1.1",
        "response_status": 200,
        "response_time_ms": 125,
        "bytes_sent": 2048,
        "referer": "http://example.com",
        "session_id": "XYZ12345",
        "user_id": "user123",
        "error_code": null,
        "error_message": null,
        "additional_info": {
            "query_params": {
                "type": "summary",
                "id": "987"
            },
            "headers": {
                "Host": "api.example.com",
                "Accept-Language": "en-US,en;q=0.8",
                "Accept-Encoding": "gzip, deflate, sdch"
            }
        }
    }
"""
    )
)
