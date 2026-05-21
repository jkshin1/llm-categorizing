from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llm_categorizing.config import LLMSettings
from llm_categorizing.knowledge import (
    DEFAULT_KNOWLEDGE_DB_PATH,
    JobKnowledgeStore,
    KnowledgeDraft,
    KnowledgeNormalizer,
    fallback_knowledge_draft,
    validate_draft_against_taxonomy,
)
from llm_categorizing.taxonomy import Taxonomy, normalize_cell


DEFAULT_TAXONOMY_PATH = "data/input/taxonomy.csv"
MAX_JSON_BODY_BYTES = 65536
MAX_IMPORT_BODY_BYTES = 1024 * 1024
MAX_IMPORT_CHUNKS = 100
MAX_IMPORT_CHUNK_CHARS = 4000

INDEX_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>직무 분류 지식 입력</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1c2430;
      --muted: #687386;
      --line: #d9dee7;
      --accent: #1d6f5f;
      --accent-strong: #15594c;
      --info: #2457a6;
      --info-bg: #eef5ff;
      --warn: #9a5b00;
      --warn-bg: #fff7e6;
      --danger: #b42318;
      --shadow: 0 1px 2px rgba(20, 27, 36, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    .shell {
      width: min(1120px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      grid-template-columns: minmax(360px, 0.92fr) minmax(420px, 1.08fr);
      gap: 16px;
      align-items: start;
    }
    header {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 40px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .status {
      color: var(--muted);
      min-width: 180px;
      text-align: right;
      white-space: nowrap;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .composer {
      padding: 16px;
      display: grid;
      gap: 12px;
    }
    .input-rules {
      border: 1px solid #f0d89b;
      border-left: 5px solid var(--warn);
      background: var(--warn-bg);
      border-radius: 6px;
      padding: 12px 14px;
      display: grid;
      gap: 10px;
    }
    .rules-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .rules-title {
      font-size: 15px;
      font-weight: 800;
      color: #633c00;
    }
    .rules-tag {
      flex-shrink: 0;
      border: 1px solid #e7c875;
      border-radius: 999px;
      color: #704600;
      font-size: 12px;
      font-weight: 800;
      padding: 2px 8px;
      background: #ffffff;
    }
    .rule-grid {
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }
    .rule-grid li {
      display: grid;
      grid-template-columns: 22px minmax(0, 1fr);
      gap: 8px;
      line-height: 1.45;
      color: #2f3540;
    }
    .rule-num {
      display: inline-grid;
      place-items: center;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: #ffffff;
      color: #704600;
      font-size: 12px;
      font-weight: 800;
      border: 1px solid #e7c875;
    }
    .rule-example {
      border-left: 4px solid var(--info);
      background: var(--info-bg);
      border-radius: 5px;
      padding: 9px 10px;
      color: #20324f;
      line-height: 1.5;
    }
    .rule-example strong {
      color: #173a70;
    }
    .rule-example code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      background: rgba(255,255,255,0.75);
      border: 1px solid #c9daf5;
      border-radius: 4px;
      padding: 1px 4px;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    textarea {
      width: 100%;
      min-height: 152px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      font: inherit;
      line-height: 1.5;
      color: var(--text);
      outline: none;
    }
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(29, 111, 95, 0.12);
    }
    .actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }
    .file-import {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
      padding: 12px;
      display: grid;
      gap: 10px;
    }
    .file-import-head {
      display: grid;
      gap: 3px;
    }
    .file-import-title {
      font-weight: 800;
      line-height: 1.35;
    }
    .file-import-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .file-import-controls {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
    }
    input[type="file"] {
      min-width: 0;
      width: 100%;
      color: var(--muted);
      font: inherit;
      font-size: 13px;
    }
    label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      user-select: none;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      font: inherit;
      font-weight: 600;
      min-height: 36px;
      padding: 0 12px;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }
    button.primary:hover { background: var(--accent-strong); }
    button.danger {
      color: var(--danger);
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    .list {
      display: grid;
      gap: 10px;
      padding: 12px;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #ffffff;
      display: grid;
      gap: 8px;
    }
    .item.inactive {
      opacity: 0.58;
    }
    .item-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }
    .title {
      font-weight: 700;
      line-height: 1.35;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .badges {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .badge {
      border-radius: 4px;
      background: #eef3f1;
      color: #31584f;
      font-size: 12px;
      font-weight: 700;
      padding: 3px 6px;
    }
    .badge.warn {
      background: #fff4e5;
      color: #9a5b00;
    }
    .badge.danger {
      background: #fef3f2;
      color: var(--danger);
    }
    .hint {
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      color: var(--muted);
      font-size: 12px;
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    .row-actions {
      display: flex;
      gap: 6px;
      flex-shrink: 0;
    }
    .empty {
      padding: 32px 16px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 860px) {
      .shell {
        grid-template-columns: 1fr;
        width: min(100vw - 24px, 680px);
        margin: 16px auto;
      }
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      .status {
        text-align: left;
        min-width: 0;
      }
      .rules-head {
        align-items: flex-start;
        flex-direction: column;
      }
      .file-import-controls {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <h1>직무 분류 지식 입력</h1>
      <div id="status" class="status"></div>
    </header>
    <section class="panel composer">
      <section class="input-rules" aria-labelledby="inputRulesTitle">
        <div class="rules-head">
          <div id="inputRulesTitle" class="rules-title">입력 시 반드시 지킬 규칙</div>
          <div class="rules-tag">분류 품질 기준</div>
        </div>
        <ul class="rule-grid">
          <li><span class="rule-num">1</span><span>용어/alias, 의미, 적용될 taxonomy 후보를 한 문장에 함께 적습니다.</span></li>
          <li><span class="rule-num">2</span><span>팀명·프로젝트명·제품명처럼 숨은 의미가 있는 단어는 표기 변형까지 적습니다.</span></li>
          <li><span class="rule-num">3</span><span>항상 맞지 않는 예외가 있으면 예외 조건을 같이 적습니다.</span></li>
          <li><span class="rule-num">4</span><span>추측, 일회성 사례, 구성원 이름·사번 같은 개인정보는 입력하지 않습니다.</span></li>
        </ul>
        <div class="rule-example">
          <strong>좋은 입력 예:</strong>
          <code>Heraion 또는 Heraion TD가 team에 있으면 NAND 제품 프로젝트를 의미한다. TD는 중직무 소자 후보를 우선 검토하되, 진단 직무명이 공정이면 진단 직무명을 우선한다.</code>
        </div>
      </section>
      <textarea id="knowledgeText" placeholder="예: 특정 팀명, 업무 표현, 산출물 표현이 나오면 어떤 taxonomy 후보를 우선 검토해야 하는지 입력"></textarea>
      <section class="file-import" aria-label="TXT 지식 일괄 가져오기">
        <div class="file-import-head">
          <div class="file-import-title">TXT 줄 단위 가져오기</div>
          <div class="file-import-note">빈 줄은 제외하고, 줄바꿈 1줄을 지식 1개로 저장합니다. 한 번에 최대 100줄, 줄당 최대 4,000자입니다.</div>
        </div>
        <div class="file-import-controls">
          <input id="txtFile" type="file" accept=".txt,text/plain">
          <button id="importBtn" type="button">TXT 가져오기</button>
        </div>
      </section>
      <div class="actions">
        <label><input id="useLlm" type="checkbox" checked> LLM 정리</label>
        <button id="saveBtn" class="primary" type="button">저장</button>
      </div>
    </section>
    <section class="panel">
      <div id="knowledgeList" class="list"></div>
    </section>
  </main>
  <script>
    const textEl = document.querySelector("#knowledgeText");
    const saveBtn = document.querySelector("#saveBtn");
    const importBtn = document.querySelector("#importBtn");
    const fileEl = document.querySelector("#txtFile");
    const listEl = document.querySelector("#knowledgeList");
    const statusEl = document.querySelector("#status");
    const useLlmEl = document.querySelector("#useLlm");

    const setStatus = (message) => {
      statusEl.textContent = message || "";
    };

    const escapeHtml = (value) => String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

    const entryTarget = (entry) => [
      entry.target_major_job,
      entry.target_sub_job,
      entry.target_device,
      entry.target_unit_job,
      entry.target_detail_job_1,
      entry.target_detail_job_2
    ].filter(Boolean).join(" > ");

    async function requestJson(url, options = {}) {
      const response = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
      }
      return payload;
    }

    async function loadEntries() {
      const payload = await requestJson("/api/knowledge");
      renderEntries(payload.items || []);
    }

    function renderEntries(items) {
      if (!items.length) {
        listEl.innerHTML = '<div class="empty">저장된 지식이 없습니다.</div>';
        return;
      }
      listEl.innerHTML = items.map((entry) => {
        const target = entryTarget(entry);
        const aliases = (entry.aliases || []).slice(0, 12)
          .map((alias) => `<span class="chip">${escapeHtml(alias)}</span>`)
          .join("");
        const validationErrors = (entry.validation_errors || [])
          .map((error) => `<span class="chip">${escapeHtml(error)}</span>`)
          .join("");
        const matchFields = (entry.match_fields || [])
          .map((field) => `<span class="chip">${escapeHtml(field)}</span>`)
          .join("");
        const conflicts = (entry.conflicts || [])
          .map((conflict) => {
            const field = conflict.field || "target";
            const existing = conflict.existing_value || "";
            const next = conflict.new_value || "";
            const title = conflict.title || conflict.knowledge_id || "";
            return `<span class="chip">${escapeHtml(field)}: ${escapeHtml(existing)} ↔ ${escapeHtml(next)} · ${escapeHtml(title)}</span>`;
          })
          .join("");
        const isVerified = entry.knowledge_type === "verified_rule" || entry.review_status === "approved";
        return `
          <article class="item ${entry.active ? "" : "inactive"}" data-id="${escapeHtml(entry.id)}">
            <div class="item-head">
              <div>
                <div class="title">${escapeHtml(entry.title)}</div>
                <div class="meta">${escapeHtml(entry.id)} · priority ${escapeHtml(entry.priority)} · confidence ${escapeHtml(entry.confidence)}</div>
              </div>
              <div class="row-actions">
                <button type="button" data-action="${isVerified ? "draft" : "approve"}">${isVerified ? "초안" : "승격"}</button>
                <button type="button" data-action="toggle">${entry.active ? "비활성" : "활성"}</button>
                <button type="button" class="danger" data-action="delete">삭제</button>
              </div>
            </div>
            <div class="badges">
              <span class="badge">${escapeHtml(entry.knowledge_type)}</span>
              <span class="badge">${escapeHtml(entry.review_status)}</span>
              ${matchFields ? '<span class="badge">적용 입력</span>' : ""}
              ${validationErrors ? '<span class="badge warn">검증 경고</span>' : ""}
              ${conflicts ? '<span class="badge danger">충돌 확인</span>' : ""}
            </div>
            <div class="hint">${escapeHtml(entry.hint)}</div>
            ${target ? `<div class="meta">${escapeHtml(target)}</div>` : ""}
            ${matchFields ? `<div class="chips">${matchFields}</div>` : ""}
            ${aliases ? `<div class="chips">${aliases}</div>` : ""}
            ${validationErrors ? `<div class="chips">${validationErrors}</div>` : ""}
            ${conflicts ? `<div class="chips">${conflicts}</div>` : ""}
          </article>
        `;
      }).join("");
    }

    async function saveEntry() {
      const text = textEl.value.trim();
      if (!text) {
        setStatus("입력 필요");
        textEl.focus();
        return;
      }
      saveBtn.disabled = true;
      setStatus("정리 중");
      try {
        await requestJson("/api/knowledge", {
          method: "POST",
          body: JSON.stringify({ text, use_llm: useLlmEl.checked })
        });
        textEl.value = "";
        setStatus("저장됨");
        await loadEntries();
      } catch (error) {
        setStatus(error.message);
      } finally {
        saveBtn.disabled = false;
      }
    }

    async function importTextFile() {
      const file = fileEl.files && fileEl.files[0];
      if (!file) {
        setStatus("TXT 파일 선택 필요");
        fileEl.focus();
        return;
      }
      importBtn.disabled = true;
      saveBtn.disabled = true;
      setStatus("TXT 읽는 중");
      try {
        const text = await file.text();
        setStatus("줄 단위 정리 중");
        const payload = await requestJson("/api/knowledge/import-text", {
          method: "POST",
          body: JSON.stringify({ text, use_llm: useLlmEl.checked })
        });
        fileEl.value = "";
        const errorSuffix = payload.errors && payload.errors.length
          ? ` · 실패 ${payload.errors.length}건`
          : "";
        setStatus(`${payload.created_count || 0}개 저장됨${errorSuffix}`);
        await loadEntries();
      } catch (error) {
        setStatus(error.message);
      } finally {
        importBtn.disabled = false;
        saveBtn.disabled = false;
      }
    }

    saveBtn.addEventListener("click", saveEntry);
    importBtn.addEventListener("click", importTextFile);
    textEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        saveEntry();
      }
    });
    listEl.addEventListener("click", async (event) => {
      const button = event.target.closest("button");
      const item = event.target.closest(".item");
      if (!button || !item) return;
      const id = item.dataset.id;
      const action = button.dataset.action;
      try {
        if (action === "toggle") {
          const active = item.classList.contains("inactive");
          await requestJson(`/api/knowledge/${id}/active`, {
            method: "POST",
            body: JSON.stringify({ active })
          });
        } else if (action === "approve") {
          await requestJson(`/api/knowledge/${id}/metadata`, {
            method: "POST",
            body: JSON.stringify({ knowledge_type: "verified_rule", review_status: "approved" })
          });
        } else if (action === "draft") {
          await requestJson(`/api/knowledge/${id}/metadata`, {
            method: "POST",
            body: JSON.stringify({ knowledge_type: "soft_hint", review_status: "draft" })
          });
        } else if (action === "delete") {
          await requestJson(`/api/knowledge/${id}`, { method: "DELETE" });
        }
        await loadEntries();
        setStatus("반영됨");
      } catch (error) {
        setStatus(error.message);
      }
    });
    loadEntries().catch((error) => setStatus(error.message));
  </script>
</body>
</html>
"""


def split_import_text(
    raw_text: str,
    *,
    max_chunks: int = MAX_IMPORT_CHUNKS,
    max_chunk_chars: int = MAX_IMPORT_CHUNK_CHARS,
) -> list[str]:
    chunks: list[str] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        chunk = normalize_cell(line)
        if not chunk:
            continue
        if len(chunk) > max_chunk_chars:
            raise ValueError(
                f"line {line_number} is too long: max {max_chunk_chars} characters"
            )
        chunks.append(chunk)
        if len(chunks) > max_chunks:
            raise ValueError(f"too many import lines: max {max_chunks}")

    if not chunks:
        raise ValueError("text has no usable non-blank lines")
    return chunks


def normalize_knowledge_text(
    raw_text: str,
    *,
    normalizer: KnowledgeNormalizer | None,
    taxonomy: Taxonomy | None,
    use_llm: bool,
    allow_fallback_normalizer: bool,
) -> KnowledgeDraft:
    try:
        if use_llm:
            if not normalizer:
                raise RuntimeError("LLM normalizer is not configured")
            draft = normalizer.normalize(raw_text)
        else:
            draft = fallback_knowledge_draft(raw_text)
    except Exception:
        if not allow_fallback_normalizer:
            raise
        draft = fallback_knowledge_draft(raw_text)

    if taxonomy:
        draft = validate_draft_against_taxonomy(draft, taxonomy)
    return draft


def import_text_chunks(
    store: JobKnowledgeStore,
    raw_text: str,
    *,
    normalizer: KnowledgeNormalizer | None,
    taxonomy: Taxonomy | None,
    use_llm: bool,
    allow_fallback_normalizer: bool,
) -> dict[str, Any]:
    chunks = split_import_text(raw_text)
    items: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks, start=1):
        try:
            draft = normalize_knowledge_text(
                chunk,
                normalizer=normalizer,
                taxonomy=taxonomy,
                use_llm=use_llm,
                allow_fallback_normalizer=allow_fallback_normalizer,
            )
            entry = store.add(chunk, draft, source="txt_import")
        except Exception as exc:
            errors.append(
                {
                    "line": index,
                    "text": chunk[:120],
                    "error": str(exc),
                }
            )
            continue
        items.append(entry.to_api_dict())

    return {
        "items": items,
        "created_count": len(items),
        "chunk_count": len(chunks),
        "errors": errors,
    }


class KnowledgeRequestHandler(BaseHTTPRequestHandler):
    store: JobKnowledgeStore
    normalizer: KnowledgeNormalizer | None
    taxonomy: Taxonomy | None
    allow_fallback_normalizer: bool

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"", "/", "/index.html"}:
            self._send_html(INDEX_HTML)
            return
        if path == "/api/knowledge":
            items = [item.to_api_dict() for item in self.store.list_recent(limit=100)]
            self._send_json({"items": items})
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/knowledge":
            self._handle_create()
            return
        if path == "/api/knowledge/import-text":
            self._handle_import_text()
            return
        if path.startswith("/api/knowledge/") and path.endswith("/active"):
            entry_id = path.removeprefix("/api/knowledge/").removesuffix("/active").strip("/")
            self._handle_set_active(entry_id)
            return
        if path.startswith("/api/knowledge/") and path.endswith("/metadata"):
            entry_id = path.removeprefix("/api/knowledge/").removesuffix("/metadata").strip("/")
            self._handle_update_metadata(entry_id)
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/knowledge/"):
            entry_id = path.removeprefix("/api/knowledge/").strip("/")
            deleted = self.store.delete(entry_id)
            self._send_json({"deleted": deleted})
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_create(self) -> None:
        try:
            payload = self._read_json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        raw_text = str(payload.get("text", "")).strip()
        use_llm = bool(payload.get("use_llm", True))
        if not raw_text:
            self._send_json({"error": "text is blank"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            draft = normalize_knowledge_text(
                raw_text,
                normalizer=self.normalizer,
                taxonomy=self.taxonomy,
                use_llm=use_llm,
                allow_fallback_normalizer=self.allow_fallback_normalizer,
            )
        except Exception as exc:
            self._send_json(
                {"error": f"knowledge normalization failed: {exc}"},
                status=HTTPStatus.BAD_GATEWAY,
            )
            return

        entry = self.store.add(raw_text, draft)
        self._send_json({"item": entry.to_api_dict()}, status=HTTPStatus.CREATED)

    def _handle_import_text(self) -> None:
        try:
            payload = self._read_json(max_bytes=MAX_IMPORT_BODY_BYTES)
            raw_text = str(payload.get("text", "")).strip()
            use_llm = bool(payload.get("use_llm", True))
            result = import_text_chunks(
                self.store,
                raw_text,
                normalizer=self.normalizer,
                taxonomy=self.taxonomy,
                use_llm=use_llm,
                allow_fallback_normalizer=self.allow_fallback_normalizer,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        status = HTTPStatus.CREATED if result["created_count"] else HTTPStatus.BAD_GATEWAY
        self._send_json(result, status=status)

    def _handle_set_active(self, entry_id: str) -> None:
        try:
            payload = self._read_json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        entry = self.store.set_active(entry_id, bool(payload.get("active", True)))
        if not entry:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json({"item": entry.to_api_dict()})

    def _handle_update_metadata(self, entry_id: str) -> None:
        try:
            payload = self._read_json()
            entry = self.store.update_metadata(
                entry_id,
                knowledge_type=payload.get("knowledge_type"),
                review_status=payload.get("review_status"),
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if not entry:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json({"item": entry.to_api_dict()})

    def _read_json(self, *, max_bytes: int = MAX_JSON_BODY_BYTES) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > max_bytes:
            raise ValueError(f"request body is too large: max {max_bytes} bytes")
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_html(self, html: str) -> None:
        encoded = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the job knowledge input page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--knowledge-db-path", default=DEFAULT_KNOWLEDGE_DB_PATH)
    parser.add_argument(
        "--taxonomy",
        default=DEFAULT_TAXONOMY_PATH,
        help=f"지식 정리 target 검증에 사용할 taxonomy CSV 경로. 기본값: {DEFAULT_TAXONOMY_PATH}",
    )
    parser.add_argument(
        "--allow-fallback-normalizer",
        action="store_true",
        help="LLM 정리 실패 시 raw 입력 기반 draft로 저장",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = JobKnowledgeStore(args.knowledge_db_path)
    taxonomy = None
    taxonomy_path = Path(args.taxonomy)
    if taxonomy_path.exists():
        taxonomy = Taxonomy.from_csv(taxonomy_path)
    else:
        print(f"Taxonomy CSV not found. Knowledge target validation disabled: {taxonomy_path}")

    normalizer = None
    try:
        normalizer = KnowledgeNormalizer(LLMSettings.from_env(), taxonomy=taxonomy)
    except Exception as exc:
        if not args.allow_fallback_normalizer:
            print(f"LLM normalizer unavailable: {exc}")
            print("Set INTERNAL_LLM_* or ALIBABA_* env vars, or run with --allow-fallback-normalizer.")
        else:
            print(f"LLM normalizer unavailable; fallback enabled: {exc}")

    KnowledgeRequestHandler.store = store
    KnowledgeRequestHandler.normalizer = normalizer
    KnowledgeRequestHandler.taxonomy = taxonomy
    KnowledgeRequestHandler.allow_fallback_normalizer = args.allow_fallback_normalizer

    server = ThreadingHTTPServer((args.host, args.port), KnowledgeRequestHandler)
    print(f"Knowledge input page: http://{args.host}:{args.port}")
    print(f"Knowledge DB: {store.path}")
    if taxonomy:
        print(f"Knowledge taxonomy validation: {taxonomy_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped knowledge input page.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
