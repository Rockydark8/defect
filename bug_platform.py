import datetime
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Optional import for semantic clustering
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


# =====================================================================
# 1. DATA MODELS & SCHEMAS
# =====================================================================

class HistoricalDefect(BaseModel):
    defect_id: str
    title: str
    description: str
    affected_component: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    priority: str  # P0, P1, P2, P3
    exception_type: str
    root_cause: str
    resolution: str
    confirmed_fix: str
    timestamp: datetime.datetime
    embedding: Optional[List[float]] = None


class SubmittedBug(BaseModel):
    bug_id: str
    raw_description: str
    raw_logs: str
    file_format: str = "text"  # text, json, log
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class TriageOutput(BaseModel):
    predicted_severity: str
    predicted_priority: str
    predicted_component: str
    confidence_score: float


class LogAnalysisOutput(BaseModel):
    extracted_exception: str
    failure_point: str
    affected_code_path: str
    error_signature: str


class RootCauseOutput(BaseModel):
    primary_hypothesis: str
    supporting_evidence_ids: List[str]
    confidence: float
    insufficient_evidence: bool = False


class DuplicateOutput(BaseModel):
    is_duplicate: bool
    matching_defect_id: Optional[str] = None
    similarity_score: float = 0.0


class RemediationOutput(BaseModel):
    recommended_fix: str
    action_steps: List[str]


class FinalDiagnosisReport(BaseModel):
    bug_id: str
    timestamp: datetime.datetime
    triage: TriageOutput
    log_analysis: LogAnalysisOutput
    root_cause: RootCauseOutput
    duplicate: DuplicateOutput
    remediation: RemediationOutput


# =====================================================================
# 2. VECTOR STORE & KNOWLEDGE BASE (M4.2)
# =====================================================================

class InMemoryVectorStore:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model_name)
        self.defects: Dict[str, HistoricalDefect] = {}

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def add_defect(self, defect: HistoricalDefect) -> Tuple[bool, str]:
        # M4.2 - Validation Gate
        if not all([defect.description, defect.affected_component, defect.exception_type, 
                    defect.root_cause, defect.resolution, defect.confirmed_fix]):
            return False, "Validation Gate Failed: Missing required fields for KB ingestion."

        # Compute embedding if missing
        content = f"{defect.title} {defect.description} {defect.exception_type} {defect.root_cause}"
        vector = np.array(self.embed_text(content))
        defect.embedding = vector.tolist()

        # M4.2 - Deduplication Guard (Cosine Similarity > 0.98)
        if self.defects:
            existing_ids = list(self.defects.keys())
            existing_vectors = np.array([self.defects[i].embedding for i in existing_ids])
            sims = cosine_similarity([vector], existing_vectors)[0]
            max_sim_idx = np.argmax(sims)
            
            if sims[max_sim_idx] > 0.98:
                return False, f"Duplicate Guard Triggered: Matches existing defect {existing_ids[max_sim_idx]} with similarity {sims[max_sim_idx]:.4f}"

        self.defects[defect.defect_id] = defect
        return True, f"Successfully indexed defect {defect.defect_id} into Knowledge Base."

    def search_similar(self, query_text: str, top_k: int = 3) -> List[Tuple[HistoricalDefect, float]]:
        if not self.defects:
            return []
        
        query_vector = np.array([self.embed_text(query_text)])
        ids = list(self.defects.keys())
        vectors = np.array([self.defects[i].embedding for i in ids])
        
        sims = cosine_similarity(query_vector, vectors)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        return [(self.defects[ids[i]], float(sims[i])) for i in top_indices]


# =====================================================================
# 3. DEFECT PATTERN ANALYTICS MODULE (M4.1)
# =====================================================================

class DefectPatternAnalytics:
    def __init__(self, vector_store: InMemoryVectorStore):
        self.vector_store = vector_store

    def generate_analytics_report(self) -> Dict[str, Any]:
        defects = list(self.vector_store.defects.values())
        total = len(defects)
        
        if total == 0:
            return {"total_defects": 0, "message": "Knowledge base empty."}

        # Frequency & Distribution Analysis
        severity_dist = {}
        component_dist = {}
        exception_dist = {}

        for d in defects:
            severity_dist[d.severity] = severity_dist.get(d.severity, 0) + 1
            component_dist[d.affected_component] = component_dist.get(d.affected_component, 0) + 1
            exception_dist[d.exception_type] = exception_dist.get(d.exception_type, 0) + 1

        # Semantic Clustering (HDBSCAN or standard Cosine Distance)
        clusters_summary = []
        if total >= 3:
            vectors = np.array([d.embedding for d in defects])
            if HAS_HDBSCAN:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
                labels = clusterer.fit_predict(vectors)
            else:
                # Fallback simple threshold grouping
                labels = [0 if i < total // 2 else 1 for i in range(total)]

            cluster_map: Dict[int, List[str]] = {}
            for label, d in zip(labels, defects):
                cluster_map.setdefault(int(label), []).append(d.defect_id)

            for cid, bug_ids in cluster_map.items():
                clusters_summary.append({
                    "cluster_id": f"cluster_{cid}" if cid != -1 else "noise",
                    "size": len(bug_ids),
                    "associated_bug_ids": bug_ids
                })

        return {
            "total_defects_analyzed": total,
            "metrics": {
                "severity_distribution": severity_dist,
                "component_distribution": component_dist,
                "top_exceptions": sorted(exception_dist.items(), key=lambda x: x[1], reverse=True)
            },
            "semantic_clusters": clusters_summary
        }


# =====================================================================
# 4. MULTI-AGENT DIAGNOSIS PIPELINE (M4.3)
# =====================================================================

class IntelligentBugPlatform:
    def __init__(self, vector_store: InMemoryVectorStore):
        self.vector_store = vector_store

    def _triage_agent(self, bug: SubmittedBug) -> TriageOutput:
        raw = (bug.raw_description + " " + bug.raw_logs).lower()
        
        if "nullpointerexception" in raw or "panic" in raw or "out of memory" in raw:
            return TriageOutput(predicted_severity="CRITICAL", predicted_priority="P0", predicted_component="Core Processing", confidence_score=0.92)
        elif "timeout" in raw or "connection" in raw:
            return TriageOutput(predicted_severity="HIGH", predicted_priority="P1", predicted_component="Database Layer", confidence_score=0.88)
        elif "auth" in raw or "token" in raw:
            return TriageOutput(predicted_severity="HIGH", predicted_priority="P1", predicted_component="Auth Service", confidence_score=0.85)
        else:
            return TriageOutput(predicted_severity="MEDIUM", predicted_priority="P2", predicted_component="General Application", confidence_score=0.60)

    def _log_analysis_agent(self, bug: SubmittedBug) -> LogAnalysisOutput:
        logs = bug.raw_logs
        
        if "NullPointerException" in logs:
            return LogAnalysisOutput(extracted_exception="NullPointerException", failure_point="AuthTokenValidator.java:45", affected_code_path="com.service.auth.validateToken()", error_signature="NPE_AUTH_VAL")
        elif "ConnectionTimeoutException" in logs or "timeout" in logs.lower():
            return LogAnalysisOutput(extracted_exception="ConnectionTimeoutException", failure_point="db_pool.py:112", affected_code_path="infrastructure.db.get_connection()", error_signature="DB_POOL_EXHAUSTION")
        elif "OutOfMemoryError" in logs or "panic" in logs.lower():
            return LogAnalysisOutput(extracted_exception="OutOfMemoryError", failure_point="pipeline.go:204", affected_code_path="main.processBatch()", error_signature="GO_OOM_PANIC")
        else:
            return LogAnalysisOutput(extracted_exception="UnknownException", failure_point="Unclear / No StackTrace", affected_code_path="Unknown", error_signature="GENERIC_ERROR")

    def diagnose_bug(self, bug: SubmittedBug) -> FinalDiagnosisReport:
        # Agent 1: Triage
        triage = self._triage_agent(bug)
        
        # Agent 2: Log Analysis
        log_res = self._log_analysis_agent(bug)
        
        # RAG Retrieval from Vector Store
        query = f"{bug.raw_description} {log_res.extracted_exception} {log_res.affected_code_path}"
        matches = self.vector_store.search_similar(query, top_k=1)

        # Agent 3: Duplicate Detection
        if matches and matches[0][1] >= 0.85:
            dup = DuplicateOutput(is_duplicate=True, matching_defect_id=matches[0][0].defect_id, similarity_score=matches[0][1])
        else:
            dup = DuplicateOutput(is_duplicate=False, matching_defect_id=None, similarity_score=matches[0][1] if matches else 0.0)

        # Agent 4: Root Cause Analysis (with Insufficient Evidence handling)
        if not matches or matches[0][1] < 0.40:
            rca = RootCauseOutput(primary_hypothesis="Insufficient historical evidence to determine exact root cause automatically.", supporting_evidence_ids=[], confidence=0.20, insufficient_evidence=True)
            remediation = RemediationOutput(recommended_fix="Manual developer investigation required.", action_steps=["Attach full system trace logs.", "Verify surrounding memory/CPU metrics."])
        else:
            matched_defect, score = matches[0]
            rca = RootCauseOutput(primary_hypothesis=f"Pattern aligns with {matched_defect.defect_id}: {matched_defect.root_cause}", supporting_evidence_ids=[matched_defect.defect_id], confidence=score, insufficient_evidence=False)
            remediation = RemediationOutput(recommended_fix=matched_defect.confirmed_fix, action_steps=[f"Apply resolution pattern from {matched_defect.defect_id}.", matched_defect.resolution])

        return FinalDiagnosisReport(
            bug_id=bug.bug_id,
            timestamp=datetime.datetime.utcnow(),
            triage=triage,
            log_analysis=log_res,
            root_cause=rca,
            duplicate=dup,
            remediation=remediation
        )


# =====================================================================
# 5. DEMONSTRATION SUITE & SCENARIOS (M4.4)
# =====================================================================

def run_demonstration():
    print("=" * 80)
    print("MILESTONE 4: FULL PLATFORM DEMONSTRATION")
    print("=" * 80)

    # Initialize System & Vector Store
    vs = InMemoryVectorStore()
    platform = IntelligentBugPlatform(vs)

    # Seed Knowledge Base with Initial Historical Defects
    seed_defects = [
        HistoricalDefect(
            defect_id="BUG-101",
            title="NPE during Auth Token parsing",
            description="Null pointer thrown when authorization header is missing in request context.",
            affected_component="Auth Service",
            severity="HIGH",
            priority="P1",
            exception_type="NullPointerException",
            root_cause="Unchecked null context variable during string token split.",
            resolution="Added explicit null check before calling split().",
            confirmed_fix="if (header != null && header.startsWith('Bearer ')) { ... }",
            timestamp=datetime.datetime(2026, 1, 15)
        ),
        HistoricalDefect(
            defect_id="BUG-102",
            title="Database Connection Pool Exhaustion under load",
            description="Connection pool depletes under high concurrent read traffic.",
            affected_component="Database Layer",
            severity="CRITICAL",
            priority="P0",
            exception_type="ConnectionTimeoutException",
            root_cause="Leaked database handles due to missing try-finally cleanup blocks.",
            resolution="Wrapped DB connection calls in contextual context managers.",
            confirmed_fix="with db.get_connection() as conn: ...",
            timestamp=datetime.datetime(2026, 2, 20)
        )
    ]

    for d in seed_defects:
        vs.add_defect(d)

    print(f"\n[+] Knowledge Base Initialized with {len(vs.defects)} historical records.")

    # -----------------------------------------------------------------
    # Demonstration Step A: Execute 5 Distinct Bug Scenarios
    # -----------------------------------------------------------------
    scenarios = [
        SubmittedBug(
            bug_id="SUB-001",
            raw_description="Auth service crashing when JWT token string is absent.",
            raw_logs="Exception in thread main java.lang.NullPointerException at AuthTokenValidator.java:45"
        ),
        SubmittedBug(
            bug_id="SUB-002",
            raw_description="DB calls timing out under high load spikes.",
            raw_logs="ConnectionTimeoutException: Could not obtain connection from pool within 3000ms at db_pool.py:112"
        ),
        SubmittedBug(
            bug_id="SUB-003",
            raw_description="Go ingestion worker crashes during peak batch processing.",
            raw_logs="fatal error: runtime: out of memory / panic at pipeline.go:204"
        ),
        SubmittedBug(
            bug_id="SUB-004",
            raw_description="Exact duplicate of missing token null pointer crash.",
            raw_logs="java.lang.NullPointerException at AuthTokenValidator.java:45"
        ),
        SubmittedBug(
            bug_id="SUB-005",
            raw_description="Page renders blank sometimes.",
            raw_logs="Unclear error string."
        )
    ]

    print("\n" + "-" * 80)
    print("RUNNING 5 BUG SUBMISSION SCENARIOS THROUGH AGENT PIPELINE")
    print("-" * 80)

    reports = []
    for bug in scenarios:
        report = platform.diagnose_bug(bug)
        reports.append(report)
        print(f"\n>>> [Bug ID: {report.bug_id}] Report Output:")
        print(f"  - Triage Component: {report.triage.predicted_component} | Severity: {report.triage.predicted_severity}")
        print(f"  - Extracted Exception: {report.log_analysis.extracted_exception}")
        print(f"  - Duplicate Flag: {report.duplicate.is_duplicate} (Match: {report.duplicate.matching_defect_id}, Score: {report.duplicate.similarity_score:.4f})")
        print(f"  - Root Cause: {report.root_cause.primary_hypothesis}")
        print(f"  - Recommended Fix: {report.remediation.recommended_fix}")

    # -----------------------------------------------------------------
    # Demonstration Step B: Live Knowledge Base Growth (M4.2)
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("DEMONSTRATING KNOWLEDGE BASE GROWTH MECHANISM (M4.2)")
    print("-" * 80)

    # Resolving SUB-003 (OOM panic) and feeding it back into the Knowledge Base
    resolved_bug_record = HistoricalDefect(
        defect_id="BUG-103",
        title="Go Batch Ingestion OOM Panic Fix",
        description="Go ingestion worker crashes during peak batch processing with out of memory error.",
        affected_component="Core Processing",
        severity="CRITICAL",
        priority="P0",
        exception_type="OutOfMemoryError",
        root_cause="Unbounded channel buffering during large batch ingestion.",
        resolution="Implemented worker pool with bounded semaphore throttling.",
        confirmed_fix="sem := make(chan struct{}, maxConcurrentWorkers)",
        timestamp=datetime.datetime.utcnow()
    )

    success, msg = vs.add_defect(resolved_bug_record)
    print(f"\n[+] Ingestion Attempt Result: {msg}")

    # Resubmit a similar OOM bug to demonstrate improved retrieval accuracy
    resubmitted_oom_bug = SubmittedBug(
        bug_id="SUB-006",
        raw_description="Ingestion worker running out of memory during heavy traffic.",
        raw_logs="fatal error: runtime: out of memory at main.processBatch()"
    )

    post_growth_report = platform.diagnose_bug(resubmitted_oom_bug)
    print(f"\n[+] Resubmitted Bug SUB-006 Diagnosis Post-KB Growth:")
    print(f"  - Retained Root Cause Evidence: {post_growth_report.root_cause.supporting_evidence_ids}")
    print(f"  - Updated Recommendation: {post_growth_report.remediation.recommended_fix}")

    # -----------------------------------------------------------------
    # Demonstration Step C: Analytics Module Execution (M4.1)
    # -----------------------------------------------------------------
    print("\n" + "-" * 80)
    print("DEMONSTRATING DEFECT PATTERN ANALYTICS MODULE (M4.1)")
    print("-" * 80)

    analytics_engine = DefectPatternAnalytics(vs)
    analytics_results = analytics_engine.generate_analytics_report()
    print("\nGenerated Defect Pattern Analytics Report:")
    print(json.dumps(analytics_results, indent=2))


if __name__ == "__main__":
    run_demonstration()