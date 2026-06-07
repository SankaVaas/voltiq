"""
agents/graph.py — LangGraph multi-agent orchestration for Voltiq.

Agent graph:
  User query
    │
    ▼
  Router agent  ─────────────────────────────────────────────┐
    │                                                         │
    ▼                                                         │
  [ingest_agent]  →  fetch & validate latest grid data        │
    │                                                         │
    ▼                                                         │
  [forecast_agent]  →  run TFT forecast, return predictions   │
    │                                                         │
    ▼                                                         │
  [anomaly_agent]  →  scan for anomalies, produce alerts      │
    │                                                         │
    ▼                                                         │
  [qa_agent]  →  RAG over incident history + context          │
    │                                                         │
    ▼                                                         │
  Synthesiser  →  merge outputs, format final response  ◄─────┘

Each agent is a LangGraph node. Edges route based on query type.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


# ── Shared state ──────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """Shared state passed between all agents in the graph."""

    messages: Annotated[list, add_messages]

    # Query metadata
    query: str
    query_type: str  # "forecast" | "anomaly" | "qa" | "general"
    country: str

    # Agent outputs
    ingest_result: dict[str, Any]
    forecast_result: dict[str, Any]
    anomaly_result: dict[str, Any]
    qa_result: dict[str, Any]

    # Final
    final_response: str
    error: str | None


# ── LLM client ───────────────────────────────────────────────────────────────


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.llm_model,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=settings.llm_max_tokens,
        temperature=0.1,
    )


# ── Agent nodes ───────────────────────────────────────────────────────────────


def router_agent(state: AgentState) -> AgentState:
    """
    Classify the operator's query and set query_type.
    Uses LLM for classification — no hardcoded rules.
    """
    llm = _get_llm()
    system = SystemMessage(
        content="""You are a routing agent for a grid analytics platform.
Classify the user query into exactly ONE of: forecast, anomaly, qa, general.
- forecast: asking for demand predictions, load forecasts, generation outlook
- anomaly: asking about outages, faults, deviations, unusual behaviour
- qa: asking about historical incidents, reports, past events
- general: anything else (greeting, help, status)

Also extract the country code if mentioned (DE, FR, ES, NL, PL). Default: DE.

Respond ONLY with JSON: {"type": "...", "country": "..."}
"""
    )
    response = llm.invoke([system, HumanMessage(content=state["query"])])

    try:
        parsed = json.loads(response.content)
        query_type = parsed.get("type", "general")
        country = parsed.get("country", "DE")
    except (json.JSONDecodeError, AttributeError):
        query_type = "general"
        country = "DE"

    logger.info("Router classified query", type=query_type, country=country)
    return {**state, "query_type": query_type, "country": country}


def ingest_agent(state: AgentState) -> AgentState:
    """Fetch latest data for the requested country."""
    from datetime import datetime, timedelta

    from data.ingest import build_feature_dataset

    try:
        end = datetime.utcnow()
        start = end - timedelta(days=7)
        df = build_feature_dataset(country=state["country"], start=start, end=end)
        result = {
            "status": "ok",
            "rows": len(df),
            "columns": df.columns.tolist(),
            "latest_timestamp": str(df["timestamp"].max()),
            "latest_load_mw": float(df["load_mw"].iloc[-1]),
        }
        logger.info("Ingest agent completed", **result)
    except Exception as e:
        result = {"status": "error", "error": str(e)}
        logger.error("Ingest agent failed", error=str(e))

    return {**state, "ingest_result": result}


def forecast_agent(state: AgentState) -> AgentState:
    """
    Run the TFT forecasting model (or fall back to a statistical baseline).
    In production this loads the MLflow-registered model.
    """
    import numpy as np

    ingest = state.get("ingest_result", {})

    if ingest.get("status") != "ok":
        return {**state, "forecast_result": {"status": "skipped", "reason": "ingest_failed"}}

    try:
        # In production: load from MLflow model registry
        # model = mlflow.pytorch.load_model("models:/voltiq_tft/Production")
        # For now: statistical baseline (mean ± trend extrapolation)
        latest_load = ingest["latest_load_mw"]
        horizon = settings.forecast_horizon

        # Simple hourly pattern simulation (replaced by TFT in production)
        hours = np.arange(horizon)
        base = latest_load
        daily_pattern = 5000 * np.sin((hours % 24 - 6) * np.pi / 12).clip(0)
        trend = -50 * (hours / 24)  # slight overnight dip
        noise = np.random.normal(0, 800, horizon)

        p50 = base + daily_pattern + trend + noise
        p10 = p50 - 3000
        p90 = p50 + 3000

        result = {
            "status": "ok",
            "country": state["country"],
            "horizon_hours": horizon,
            "p10": p10.tolist(),
            "p50": p50.tolist(),
            "p90": p90.tolist(),
            "peak_forecast_mw": float(p90.max()),
            "model": "statistical_baseline",  # replace with "tft_v1" in production
        }
        logger.info("Forecast agent completed", peak=result["peak_forecast_mw"])
    except Exception as e:
        result = {"status": "error", "error": str(e)}
        logger.error("Forecast agent failed", error=str(e))

    return {**state, "forecast_result": result}


def anomaly_agent(state: AgentState) -> AgentState:
    """Run LSTM Autoencoder anomaly detection on recent load data."""
    from pathlib import Path

    import numpy as np

    from anomaly.detector import AnomalyDetector

    try:
        # Synthetic recent series for demo (replace with real data from ingest)
        rng = np.random.default_rng(42)
        series = rng.normal(50_000, 2_000, 168)

        # Inject a synthetic anomaly
        series[72:76] = series[72:76] * 1.35

        model_path = Path(settings.model_artifact_dir) / "anomaly_detector.pt"
        detector = AnomalyDetector(model_path=model_path if model_path.exists() else None)

        if detector.threshold is None:
            # Quick calibration on the series itself (demo mode)
            detector.train(series, epochs=5, batch_size=32)

        detection = detector.detect(series)
        result = {
            "status": "ok",
            "anomaly_count": len(detection["anomaly_indices"]),
            "anomaly_indices": detection["anomaly_indices"][:20],
            "anomaly_rate": round(detection["anomaly_rate"], 4),
            "threshold": detection["threshold"],
            "severity": "high" if detection["anomaly_rate"] > 0.05 else "low",
        }
        logger.info("Anomaly agent completed", count=result["anomaly_count"])
    except Exception as e:
        result = {"status": "error", "error": str(e)}
        logger.error("Anomaly agent failed", error=str(e))

    return {**state, "anomaly_result": result}


def qa_agent(state: AgentState) -> AgentState:
    """Retrieve from vector store and generate grounded answer."""
    try:
        from rag.pipeline import GridRAGChain

        chain = GridRAGChain()
        answer_dict = chain.invoke_with_sources(state["query"])
        result = {
            "status": "ok",
            "answer": answer_dict["answer"],
            "source_count": len(answer_dict["sources"]),
        }
    except Exception as e:
        # Fall back to LLM-only if Qdrant is not running
        llm = _get_llm()
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a grid operations assistant. Answer based on general knowledge."
                ),
                HumanMessage(content=str(state["query"])),
            ]
        )
        result = {
            "status": "fallback",
            "answer": response.content,
            "source_count": 0,
            "warning": str(e),
        }
        logger.warning("QA agent fell back to LLM-only", error=str(e))

    return {**state, "qa_result": result}


def synthesiser(state: AgentState) -> AgentState:
    """
    Merge outputs from all agents into a coherent operator-facing response.
    Uses LLM to write the final summary in natural language.
    """
    llm = _get_llm()

    context_parts = [f"Operator query: {state['query']}"]

    if fc := state.get("forecast_result", {}):
        if fc.get("status") == "ok":
            context_parts.append(
                f"Forecast: {fc['horizon_hours']}h outlook for {fc['country']}. "
                f"Peak expected: {fc['peak_forecast_mw']:,.0f} MW."
            )

    if an := state.get("anomaly_result", {}):
        if an.get("status") == "ok":
            context_parts.append(
                f"Anomaly detection: {an['anomaly_count']} anomalous windows detected "
                f"(rate: {an['anomaly_rate']:.1%}, severity: {an['severity']})."
            )

    if qa := state.get("qa_result", {}):
        if qa.get("status") in ("ok", "fallback"):
            context_parts.append(f"Historical context: {qa['answer'][:800]}")

    system = SystemMessage(
        content="""You are Voltiq, an expert grid analytics assistant.
Synthesise the analysis results into a clear, professional briefing for a grid operator.
Use bullet points for key findings. Highlight any high-severity issues prominently.
Be concise — operators are busy. Max 300 words."""
    )

    human = HumanMessage(content="\n\n".join(context_parts))
    response = llm.invoke([system, human])

    return {**state, "final_response": response.content}


# ── Routing edges ─────────────────────────────────────────────────────────────


def route_after_router(state: AgentState) -> str:
    return state.get("query_type", "general")


def should_run_anomaly(state: AgentState) -> str:
    """Always run anomaly after forecast for a complete picture."""
    return "anomaly_agent"


def should_run_qa(state: AgentState) -> str:
    qtype = state.get("query_type", "general")
    if qtype in ("qa", "anomaly", "general"):
        return "qa_agent"
    return "synthesiser"


# ── Build graph ───────────────────────────────────────────────────────────────


def build_graph() -> CompiledGraph:
    """Compile and return the Voltiq LangGraph agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("router", router_agent)
    graph.add_node("ingest_agent", ingest_agent)
    graph.add_node("forecast_agent", forecast_agent)
    graph.add_node("anomaly_agent", anomaly_agent)
    graph.add_node("qa_agent", qa_agent)
    graph.add_node("synthesiser", synthesiser)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "forecast": "ingest_agent",
            "anomaly": "ingest_agent",
            "qa": "qa_agent",
            "general": "qa_agent",
        },
    )

    graph.add_edge("ingest_agent", "forecast_agent")
    graph.add_edge("forecast_agent", "anomaly_agent")
    graph.add_edge("anomaly_agent", "synthesiser")
    graph.add_edge("qa_agent", "synthesiser")
    graph.add_edge("synthesiser", END)

    return graph.compile()


# Module-level compiled graph (singleton)
voltiq_graph = build_graph()


async def run_agent(query: str, country: str = "DE") -> dict[str, Any]:
    """Entry point for the API layer."""
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "query_type": "",
        "country": country,
        "ingest_result": {},
        "forecast_result": {},
        "anomaly_result": {},
        "qa_result": {},
        "final_response": "",
        "error": None,
    }
    final_state = await voltiq_graph.ainvoke(initial_state)
    return {
        "response": final_state["final_response"],
        "query_type": final_state["query_type"],
        "country": final_state["country"],
        "forecast": final_state.get("forecast_result"),
        "anomalies": final_state.get("anomaly_result"),
    }
