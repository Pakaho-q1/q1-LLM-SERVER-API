# Production Roadmap — Local LLM Server API

เอกสารนี้สรุปแผนยกระดับโปรเจกจากระดับใช้งาน local/dev ไปสู่ production-grade แบบเป็นเฟสชัดเจน
โดยเน้น 3 แกนหลัก: Security, Reliability, Observability

---

## เป้าหมายภาพรวม (North Star)

- ปลอดภัยพอสำหรับใช้งานจริง (secure-by-default)
- เสถียรเมื่อมี concurrent users และงาน background
- ตรวจสอบปัญหาได้เร็วผ่าน metrics/logs/traces
- มี API contract ชัดเจน ลดการแตกพังฝั่ง client

---

## Scoring baseline (ปัจจุบัน)

- Architecture: 8/10
- Security: 5/10
- Reliability: 6.5/10
- Observability: 5.5/10
- Scalability: 6/10
- DX: 7.5/10
- **Overall: 6.8/10**

---

## P0 — ต้องทำก่อนขึ้น production (1–2 สัปดาห์)

### 1) Security hardening (บังคับ)
- [ ] บังคับ auth ใน production env (ห้ามปล่อยว่าง)
- [ ] แยก read/write scopes ของ API key หรือเปลี่ยนเป็น JWT
- [ ] ปิด error leakage: response กลับผู้ใช้เป็น generic message, รายละเอียดเก็บใน server logs
- [ ] เพิ่ม input validation แบบ strict ด้วย Pydantic models (chat/action/model ops)
- [ ] เพิ่ม path safety สำหรับ model load/delete (resolve path + enforce parent directory)

### 2) API contract stability
- [ ] แทนที่ `Dict[str, Any]` ด้วย request/response schemas
- [ ] ใส่ enum สำหรับ action names
- [ ] เพิ่ม compatibility layer (versioned API: `/v1/...`)

### 3) Runtime guards
- [ ] Timeouts สำหรับ downstream งานหนัก (model load, memory search)
- [ ] จำกัด queue size ต่อ client สำหรับ SSE
- [ ] ป้องกัน client_id ชนกัน/โดน hijack (server-issued token/session)

### Exit criteria (P0)
- [ ] ไม่มี unauthenticated access ใน prod
- [ ] ไม่มี path traversal ใน file ops
- [ ] 95% ของ endpoints มี typed schema
- [ ] Error responses ไม่เผยข้อมูลภายใน

---

## P1 — เสถียรภาพและการสังเกตระบบ (2–6 สัปดาห์)

### 1) Observability เต็มรูปแบบ
- [ ] Structured logging (JSON logs)
- [ ] Correlation ID / Request ID ครอบคลุมทั้ง HTTP + background task
- [ ] Metrics: request rate, error rate, p95 latency, token/sec, queue depth, active downloads
- [ ] Tracing (OpenTelemetry)

### 2) Reliability engineering
- [ ] Graceful shutdown สำหรับ in-flight chat/download task
- [ ] Retry policy พร้อม backoff สำหรับ network ops (HF fetch/download)
- [ ] Circuit breaker สำหรับ external dependency (embedding/vector store)
- [ ] Health endpoint แยก liveness/readiness

### 3) Storage & data safety
- [ ] DB migration strategy (เช่น Alembic pattern)
- [ ] Backup/restore workflow สำหรับ chat.db/presets.db และ memory store
- [ ] Data retention policy + cleanup jobs

### Exit criteria (P1)
- [ ] มี dashboard หลักพร้อม alert
- [ ] MTTR ลดลง (ตั้งเป้า < 30 นาที)
- [ ] ผ่าน load test พื้นฐานตาม SLO ที่กำหนด

---

## P2 — Scale และมาตรฐานระดับองค์กร (6–12+ สัปดาห์)

### 1) Service decomposition
- [ ] แยก API gateway / orchestration / model runtime / memory service
- [ ] ย้าย state สำคัญออกจาก process memory
- [ ] ใช้ distributed queue สำหรับงาน async

### 2) Enterprise security
- [ ] Secret management (Vault/KMS)
- [ ] Internal auth (mTLS/service identity)
- [ ] Audit logging ครอบคลุม action สำคัญ
- [ ] Policy enforcement (RBAC/ABAC)

### 3) Delivery excellence
- [ ] Canary / blue-green deployment
- [ ] SLO + error budget
- [ ] Chaos testing สำหรับ resilience

### Exit criteria (P2)
- [ ] รองรับ scale-out ได้โดยไม่ผูกกับ single-node state
- [ ] ผ่าน security review / pen test
- [ ] deploy ได้แบบ zero/low downtime

---

## Recommended PR sequence

### PR-1 (P0): Auth + Path Safety + Error Hygiene
- บังคับ auth ใน production
- sanitize path operations
- ปรับ error handling ไม่ให้ leak detail

### PR-2 (P0): Typed API Contracts
- ใส่ Pydantic models สำหรับทุก endpoint สำคัญ
- ใส่ enum สำหรับ action

### PR-3 (P1): Observability Foundation
- structured logs + request ID
- metrics + tracing hooks

### PR-4 (P1): Task Supervision & Shutdown
- background task registry
- cancellation + graceful drain

### PR-5 (P1/P2): Storage Hardening
- migrations + retention + backup scripts

---

## KPI / SLO ที่ควรกำหนด

- API success rate ≥ 99.5%
- p95 latency (`/sse/chat` first token) ≤ 2.5s (ขึ้นกับ model)
- p95 latency (`/api/action`) ≤ 400ms (non-LLM actions)
- Failed background jobs < 1%
- Mean time to detect (MTTD) < 5 นาที
- Mean time to recover (MTTR) < 30 นาที

---

## Quick checklist for next sprint

- [ ] Lock production auth behavior
- [ ] Path traversal tests for model operations
- [ ] Convert top 5 payloads to Pydantic schemas
- [ ] Add request_id + structured logs
- [ ] Add readiness endpoint and basic alerts

