"""Custom application exceptions with a shared JSON error response.

Raise ``AppException`` (or a subclass) anywhere in the app and the FastAPI
handler registered in ``app.main`` turns it into a consistent
``{"success": false, "message": ..., "data": null}`` JSON response with the
given HTTP status code.

Note: this module must NOT import ``app.main`` (that would be circular) — the
handler is registered on the app explicitly in ``app.main``.
"""

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _error_response(status_code: int, message: str) -> JSONResponse:
    """The standard API error body: {"success": false, "message": ..., "data": null}."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "data": None,
        },
    )


class AppException(Exception):
    """Base exception carrying an HTTP status code and a user-facing message."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Convert an AppException into the standard API error response."""
    return _error_response(exc.status_code, exc.message)


# ---------------------------------------------------------------------------
# Friendly 422 messages
# ---------------------------------------------------------------------------

# pydantic v2 error `type` -> human-friendly wording. `{ctx_key}` placeholders
# are filled from the error's `ctx` when available.
_FRIENDLY_TYPES: dict[str, str] = {
    "missing": "is required (must be filled)",
    "float_parsing": "must be a number",
    "int_parsing": "must be a whole number",
    "number_parsing": "must be a number",
    "decimal_parsing": "must be a number",
    "finite_number": "must be a finite number",
    "string_type": "must be text",
    "bool_parsing": "must be true or false",
    "bool_type": "must be true or false",
    "list_type": "must be a list",
    "dict_type": "must be an object",
    "model_attributes_type": "must be an object",
    "less_than": "must be less than {lt}",
    "less_than_equal": "must be less than or equal to {le}",
    "greater_than": "must be greater than {gt}",
    "greater_than_equal": "must be greater than or equal to {ge}",
    "multiple_of": "must be a multiple of {multiple_of}",
    "enum": "is not an allowed value",
    "literal_error": "is not an allowed value",
    "date_parsing": "must be a valid date",
    "time_parsing": "must be a valid time",
    "datetime_parsing": "must be a valid date-time",
    "url_parsing": "must be a valid URL",
    "email": "must be a valid email",
    "uuid_parsing": "must be a valid UUID",
    "string_too_short": "is too short (min length {min_length})",
    "string_too_long": "is too long (max length {max_length})",
}

# Fields that represent geographic coordinates. Number errors on these read
# better as "must be a coordinate", and the known ranges let us say exactly
# what a valid coordinate is (matches the Field ge=/le= constraints).
_COORDINATE_RANGES: dict[str, tuple[float, float]] = {
    "lat": (-90.0, 90.0),
    "lng": (-180.0, 180.0),
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
}

_NUMBER_TYPES = {
    "float_parsing",
    "int_parsing",
    "number_parsing",
    "decimal_parsing",
    "finite_number",
}

_RANGE_TYPES = {"less_than", "less_than_equal", "greater_than", "greater_than_equal"}


def _field_name(err: dict[str, Any]) -> str:
    """'body.origin.lat' -> 'origin.lat' (drop the body segment + list indexes)."""
    parts = [str(p) for p in err.get("loc", ()) if not isinstance(p, int)]
    if parts and parts[0] == "body":
        parts = parts[1:]
    return ".".join(parts)


def _friendly_message(err: dict[str, Any]) -> str:
    """Turn a single pydantic error into a short, human-readable message."""
    err_type = err.get("type", "")
    ctx = err.get("ctx") or {}

    # Malformed JSON has no meaningful field — report the body itself.
    if err_type == "json_invalid":
        return "Request body is not valid JSON"

    field = _field_name(err)
    leaf = field.rsplit(".", 1)[-1]

    # Coordinate fields: say "must be a coordinate" with the valid range.
    if leaf in _COORDINATE_RANGES:
        lo, hi = _COORDINATE_RANGES[leaf]
        if err_type in _NUMBER_TYPES:
            return f"Field '{field}' must be a coordinate (a number between {lo:g} and {hi:g})"
        if err_type in _RANGE_TYPES:
            return f"Field '{field}' must be a coordinate between {lo:g} and {hi:g}"

    template = _FRIENDLY_TYPES.get(err_type)
    if template:
        try:
            detail = template.format(**ctx)
        except (KeyError, ValueError):
            detail = template
    else:
        detail = err.get("msg", "is invalid")

    if not field:
        return detail.capitalize()
    return f"Field '{field}' {detail}"


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert FastAPI's 422 request-validation errors into the standard shape.

    Raw pydantic output ("Input should be less than or equal to 90") is
    rewritten into plain wording ("Field 'origin.lat' must be a coordinate
    between -90 and 90"). A single error stands alone; several are joined
    under a "Request validation error:" prefix.
    """
    errors = exc.errors()
    details = "; ".join(_friendly_message(err) for err in errors)
    if not details:
        message = "Request validation error"
    elif len(errors) == 1:
        message = details
    else:
        message = f"Request validation error: {details}"
    return _error_response(422, message)