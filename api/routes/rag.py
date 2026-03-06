from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import verify_api_key
from api.runtime import app_state
from api.schemas import RAGCitation, RAGQueryRequest, RAGRetrieveResponse

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    content = await file.read()
    result = app_state.ingestion_service.ingest_bytes(
        filename=file.filename,
        content=content,
        source_type="upload",
        metadata={"mime_type": file.content_type or "application/octet-stream"},
    )

    return {
        "doc_id": result.doc_id,
        "source_name": result.source_name,
        "source_type": result.source_type,
        "chunks": result.chunks,
    }


@router.post("/rag/retrieve", response_model=RAGRetrieveResponse)
async def rag_retrieve(payload: RAGQueryRequest):
    chunks = app_state.retrieval_service.retrieve(
        payload.query,
        limit=payload.top_k,
        use_hybrid=payload.hybrid,
        use_rerank=payload.rerank,
        score_threshold=payload.score_threshold,
    )
    grounded = app_state.retrieval_service.build_grounded_prompt(payload.query, chunks)

    citations = [RAGCitation(**c) for c in grounded.get("citations", [])]

    return RAGRetrieveResponse(
        query=payload.query,
        context=grounded.get("context", ""),
        citations=citations,
        should_answer=grounded.get("should_answer", True),
        reason=grounded.get("reason"),
    )
