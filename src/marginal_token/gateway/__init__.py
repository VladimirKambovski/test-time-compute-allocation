"""FastAPI live serving endpoint. Calls the SAME Controller instance used by replay. Three outcomes: answered / escalated / declined."""

from marginal_token.gateway.app import SolveRequest, SolveResponse, create_app

__all__ = ["SolveRequest", "SolveResponse", "create_app"]
