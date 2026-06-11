import json
from pathlib import Path
from typing import Dict, Optional, Union

import urllib3


def fetch(endpoint: str, headers: Dict, params: Optional[Dict] = None) -> Dict:
    """Fetches data from the API endpoint.

    Args:
        endpoint: API endpoint URL
        params: Optional query parameters

    Returns:
        Dict: JSON response from the API

    Raises:
        urllib3.exceptions.HTTPError: On non-200 status code
        json.JSONDecodeError: On invalid JSON response
    """
    params = {} if params is None else params
    http = urllib3.PoolManager(retries=urllib3.Retry(3))
    try:
        response = http.request("GET", endpoint, fields=params, headers=headers)
        if response.status != 200:
            error_details = response.data.decode("utf-8")
            raise urllib3.exceptions.HTTPError(f"Request failed with status {response.status}: {error_details}")

        return json.loads(response.data.decode("utf-8"))
    finally:
        http.clear()


def post(endpoint: str, headers: Dict, data: Union[Path, Dict, bytes, str], multipart: bool = False) -> Dict:
    """Posts data to backend endpoint.

    Args:
        endpoint: The URL endpoint to post data to
        data: The data to post - either a file path, dictionary, or raw bytes
        multipart: Whether to send as multipart/form-data

    Returns:
        Dict[str, Any]: The JSON response from the endpoint

    Raises:
        FileNotFoundError: If the provided Path doesn't exist
        urllib3.exceptions.HTTPError: If the request fails
        json.JSONDecodeError: If response isn't valid JSON
        ValueError: If data type is unsupported
    """
    http = urllib3.PoolManager(retries=urllib3.Retry(3))
    try:
        if multipart:
            # Remove content-type header if present as urllib3 will set it
            headers.pop("Content-Type", None)
            response = http.request(
                "POST",
                endpoint,
                fields=data if isinstance(data, dict) else {"file": data},
                headers=headers,
            )
        else:
            if isinstance(data, Path):
                with open(data, "rb") as f:
                    body = f.read()
                response = http.request("POST", endpoint, body=body, headers=headers)
            elif isinstance(data, (str, dict)):
                if isinstance(data, dict):
                    data = json.dumps(data)
                headers["Content-Type"] = "application/json"
                response = http.request("POST", endpoint, body=data, headers=headers)
            elif isinstance(data, bytes):
                response = http.request("POST", endpoint, body=data, headers=headers)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

        if response.status not in (200, 201):
            error_details = response.data.decode("utf-8")
            raise urllib3.exceptions.HTTPError(f"Request failed with status {response.status}: {error_details}")

        return json.loads(response.data.decode("utf-8"))
    finally:
        http.clear()


def patch(endpoint: str, headers: Dict, data: Union[Dict, bytes, str], multipart: bool = False) -> Dict:
    """Sends a PATCH request to the specified endpoint.

    Mirrors `post`: supports JSON dict/str/bytes payloads or multipart/form-data.
    """
    http = urllib3.PoolManager(retries=urllib3.Retry(3))
    try:
        if multipart:
            headers.pop("Content-Type", None)
            response = http.request(
                "PATCH",
                endpoint,
                fields=data if isinstance(data, dict) else {"file": data},
                headers=headers,
            )
        else:
            if isinstance(data, dict):
                data = json.dumps(data)
            if isinstance(data, str):
                headers["Content-Type"] = "application/json"
                response = http.request("PATCH", endpoint, body=data, headers=headers)
            elif isinstance(data, bytes):
                response = http.request("PATCH", endpoint, body=data, headers=headers)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

        if response.status not in (200, 202):
            error_details = response.data.decode("utf-8")
            raise urllib3.exceptions.HTTPError(f"Request failed with status {response.status}: {error_details}")

        return json.loads(response.data.decode("utf-8"))
    finally:
        http.clear()


def delete(endpoint: str, headers: Dict) -> Dict:
    """Sends a DELETE request to the specified endpoint."""
    http = urllib3.PoolManager(retries=urllib3.Retry(3))
    try:
        response = http.request("DELETE", endpoint, headers=headers)

        if response.status not in (200, 204):
            error_details = response.data.decode("utf-8")
            raise urllib3.exceptions.HTTPError(f"Request failed with status {response.status}: {error_details}")
        return json.loads(response.data.decode("utf-8")) if response.data else {}
    finally:
        http.clear()
