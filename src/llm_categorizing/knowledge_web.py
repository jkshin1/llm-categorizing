from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

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
    input[type="text"],
    input[type="number"],
    select {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      font: inherit;
      min-height: 36px;
      padding: 0 10px;
      outline: none;
    }
    input[type="text"]:focus,
    input[type="number"]:focus,
    select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(29, 111, 95, 0.12);
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
    button.subtle {
      color: var(--muted);
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    .management {
      padding: 12px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }
    .management-grid {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) 140px 120px 140px;
      gap: 8px;
    }
    .management-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
    }
    .import-ndjson {
      display: flex;
      gap: 8px;
      align-items: center;
      min-width: min(100%, 360px);
    }
    .export-actions {
      display: flex;
      gap: 8px;
      align-items: center;
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
    .badge.rule {
      background: #eef5ff;
      color: #2457a6;
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
    .modal {
      position: fixed;
      inset: 0;
      background: rgba(17, 24, 39, 0.42);
      display: grid;
      place-items: center;
      padding: 20px;
      z-index: 20;
    }
    .modal[hidden] {
      display: none;
    }
    .dialog {
      width: min(1040px, 100%);
      max-height: min(92vh, 980px);
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 16px 50px rgba(17, 24, 39, 0.24);
      display: grid;
      gap: 0;
    }
    .dialog-head {
      position: sticky;
      top: 0;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 14px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      z-index: 1;
    }
    .dialog-title {
      min-width: 0;
      font-weight: 800;
      overflow-wrap: anywhere;
    }
    .editor {
      padding: 16px;
      display: grid;
      gap: 14px;
    }
    .editor-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .field {
      display: grid;
      gap: 6px;
      min-width: 0;
    }
    .field.full {
      grid-column: 1 / -1;
    }
    .field span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .field textarea {
      min-height: 88px;
    }
    .field textarea.tall {
      min-height: 132px;
    }
    .editor-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: space-between;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    .side-panels {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
    }
    .subpanel {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      display: grid;
      gap: 8px;
      min-width: 0;
    }
    .subpanel-title {
      font-weight: 800;
    }
    .revision-row,
    .usage-row {
      border-top: 1px solid var(--line);
      padding-top: 8px;
      display: grid;
      gap: 5px;
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
      .management-grid,
      .editor-grid,
      .side-panels {
        grid-template-columns: 1fr;
      }
      .management-actions,
      .import-ndjson,
      .export-actions {
        align-items: stretch;
        flex-direction: column;
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
      <div class="management">
        <div class="management-grid">
          <input id="searchInput" type="text" placeholder="제목, alias, target 검색">
          <select id="statusFilter">
            <option value="">상태 전체</option>
            <option value="draft">draft</option>
            <option value="approved">approved</option>
            <option value="rejected">rejected</option>
          </select>
          <select id="activeFilter">
            <option value="">활성 전체</option>
            <option value="active">활성</option>
            <option value="inactive">비활성</option>
          </select>
          <select id="enforcementFilter">
            <option value="">강도 전체</option>
            <option value="soft">soft</option>
            <option value="strong">strong</option>
            <option value="near_hard">near_hard</option>
          </select>
        </div>
        <div class="management-actions">
          <div class="export-actions">
            <select id="exportScope">
              <option value="approved">approved export</option>
              <option value="usable">usable export</option>
              <option value="all">all export</option>
            </select>
            <button id="exportBtn" type="button">NDJSON 내보내기</button>
          </div>
          <div class="import-ndjson">
            <input id="ndjsonFile" type="file" accept=".ndjson,.jsonl,application/x-ndjson,text/plain">
            <button id="importNdjsonBtn" type="button">NDJSON 가져오기</button>
          </div>
        </div>
      </div>
      <div id="knowledgeList" class="list"></div>
    </section>
  </main>
  <div id="editorModal" class="modal" hidden>
    <section class="dialog" role="dialog" aria-modal="true" aria-labelledby="editorTitle">
      <div class="dialog-head">
        <div>
          <div id="editorTitle" class="dialog-title">지식 편집</div>
          <div id="editorMeta" class="meta"></div>
        </div>
        <button id="closeEditorBtn" class="subtle" type="button">닫기</button>
      </div>
      <form id="editorForm" class="editor">
        <div class="editor-grid">
          <label class="field full"><span>원문</span><textarea id="editRawText" class="tall"></textarea></label>
          <label class="field"><span>제목</span><input id="editTitle" type="text"></label>
          <label class="field"><span>knowledge_type</span><select id="editKnowledgeType">
            <option value="glossary">glossary</option>
            <option value="soft_hint">soft_hint</option>
            <option value="negative_hint">negative_hint</option>
            <option value="correction">correction</option>
            <option value="verified_rule">verified_rule</option>
          </select></label>
          <label class="field"><span>review_status</span><select id="editReviewStatus">
            <option value="draft">draft</option>
            <option value="approved">approved</option>
            <option value="rejected">rejected</option>
          </select></label>
          <label class="field"><span>enforcement_level</span><select id="editEnforcement">
            <option value="soft">soft</option>
            <option value="strong">strong</option>
            <option value="near_hard">near_hard</option>
          </select></label>
          <label class="field"><span>priority</span><input id="editPriority" type="number" min="1" max="100"></label>
          <label class="field"><span>confidence</span><input id="editConfidence" type="number" min="0" max="1" step="0.01"></label>
          <label class="field"><span>활성</span><select id="editActive">
            <option value="true">활성</option>
            <option value="false">비활성</option>
          </select></label>
          <label class="field"><span>aliases</span><textarea id="editAliases"></textarea></label>
          <label class="field"><span>match_fields</span><textarea id="editMatchFields"></textarea></label>
          <label class="field full"><span>적용 조건</span><textarea id="editAppliesWhen"></textarea></label>
          <label class="field full"><span>힌트</span><textarea id="editHint" class="tall"></textarea></label>
          <label class="field"><span>중직무</span><input id="editMajorJob" type="text"></label>
          <label class="field"><span>소직무</span><input id="editSubJob" type="text"></label>
          <label class="field"><span>Device</span><input id="editDevice" type="text"></label>
          <label class="field"><span>단위 직무</span><input id="editUnitJob" type="text"></label>
          <label class="field"><span>세부 직무1</span><input id="editDetailJob1" type="text"></label>
          <label class="field"><span>세부 직무2</span><input id="editDetailJob2" type="text"></label>
        </div>
        <div class="editor-actions">
          <label><input id="editClearConflicts" type="checkbox"> 충돌 경고 해제</label>
          <div class="row-actions">
            <button id="refreshDetailBtn" type="button">새로고침</button>
            <button id="saveEditBtn" class="primary" type="submit">저장</button>
          </div>
        </div>
        <div class="side-panels">
          <section class="subpanel">
            <div class="subpanel-title">사용 통계</div>
            <div id="usagePanel" class="meta"></div>
          </section>
          <section class="subpanel">
            <div class="subpanel-title">변경 이력</div>
            <div id="revisionPanel" class="meta"></div>
          </section>
        </div>
      </form>
    </section>
  </div>
  <script>
    const textEl = document.querySelector("#knowledgeText");
    const saveBtn = document.querySelector("#saveBtn");
    const importBtn = document.querySelector("#importBtn");
    const fileEl = document.querySelector("#txtFile");
    const listEl = document.querySelector("#knowledgeList");
    const statusEl = document.querySelector("#status");
    const useLlmEl = document.querySelector("#useLlm");
    const searchInput = document.querySelector("#searchInput");
    const statusFilter = document.querySelector("#statusFilter");
    const activeFilter = document.querySelector("#activeFilter");
    const enforcementFilter = document.querySelector("#enforcementFilter");
    const exportScope = document.querySelector("#exportScope");
    const exportBtn = document.querySelector("#exportBtn");
    const ndjsonFileEl = document.querySelector("#ndjsonFile");
    const importNdjsonBtn = document.querySelector("#importNdjsonBtn");
    const editorModal = document.querySelector("#editorModal");
    const editorForm = document.querySelector("#editorForm");
    const closeEditorBtn = document.querySelector("#closeEditorBtn");
    const refreshDetailBtn = document.querySelector("#refreshDetailBtn");
    const editorTitle = document.querySelector("#editorTitle");
    const editorMeta = document.querySelector("#editorMeta");
    const usagePanel = document.querySelector("#usagePanel");
    const revisionPanel = document.querySelector("#revisionPanel");
    const editFields = {
      raw_text: document.querySelector("#editRawText"),
      title: document.querySelector("#editTitle"),
      knowledge_type: document.querySelector("#editKnowledgeType"),
      review_status: document.querySelector("#editReviewStatus"),
      enforcement_level: document.querySelector("#editEnforcement"),
      priority: document.querySelector("#editPriority"),
      confidence: document.querySelector("#editConfidence"),
      active: document.querySelector("#editActive"),
      aliases: document.querySelector("#editAliases"),
      match_fields: document.querySelector("#editMatchFields"),
      applies_when: document.querySelector("#editAppliesWhen"),
      hint: document.querySelector("#editHint"),
      target_major_job: document.querySelector("#editMajorJob"),
      target_sub_job: document.querySelector("#editSubJob"),
      target_device: document.querySelector("#editDevice"),
      target_unit_job: document.querySelector("#editUnitJob"),
      target_detail_job_1: document.querySelector("#editDetailJob1"),
      target_detail_job_2: document.querySelector("#editDetailJob2"),
      clear_conflicts: document.querySelector("#editClearConflicts")
    };
    let knowledgeItems = [];
    let selectedEntryId = "";

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

    const splitValues = (value) => {
      const result = [];
      const seen = new Set();
      String(value || "").split(/\\r?\\n|,/).forEach((item) => {
        const text = item.trim();
        const key = text.toLocaleLowerCase();
        if (!text || seen.has(key)) return;
        seen.add(key);
        result.push(text);
      });
      return result;
    };

    const valueText = (items) => (items || []).join("\\n");

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
      knowledgeItems = payload.items || [];
      renderEntries();
    }

    function filterEntries(items) {
      const query = searchInput.value.trim().toLocaleLowerCase();
      const status = statusFilter.value;
      const active = activeFilter.value;
      const enforcement = enforcementFilter.value;
      return items.filter((entry) => {
        if (status && entry.review_status !== status) return false;
        if (enforcement && entry.enforcement_level !== enforcement) return false;
        if (active === "active" && !entry.active) return false;
        if (active === "inactive" && entry.active) return false;
        if (!query) return true;
        const searchable = [
          entry.id,
          entry.raw_text,
          entry.title,
          entry.hint,
          entry.applies_when,
          entryTarget(entry),
          ...(entry.aliases || []),
          ...(entry.match_fields || []),
          ...(entry.validation_errors || [])
        ].join(" ").toLocaleLowerCase();
        return searchable.includes(query);
      });
    }

    function renderEntries(items = knowledgeItems) {
      const displayItems = filterEntries(items);
      if (!displayItems.length) {
        listEl.innerHTML = '<div class="empty">저장된 지식이 없습니다.</div>';
        return;
      }
      listEl.innerHTML = displayItems.map((entry) => {
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
        const isNearHard = entry.enforcement_level === "near_hard";
        return `
          <article class="item ${entry.active ? "" : "inactive"}" data-id="${escapeHtml(entry.id)}">
            <div class="item-head">
              <div>
                <div class="title">${escapeHtml(entry.title)}</div>
                <div class="meta">${escapeHtml(entry.id)} · priority ${escapeHtml(entry.priority)} · confidence ${escapeHtml(entry.confidence)}</div>
              </div>
              <div class="row-actions">
                <button type="button" data-action="edit">편집</button>
                <button type="button" data-action="${isVerified ? "draft" : "approve"}">${isVerified ? "초안" : "승격"}</button>
                <button type="button" data-action="${isNearHard ? "strong-rule" : "near-hard-rule"}">${isNearHard ? "준하드 해제" : "준하드룰"}</button>
                ${conflicts ? '<button type="button" data-action="clear-conflicts">충돌 해제</button>' : ""}
                <button type="button" data-action="toggle">${entry.active ? "비활성" : "활성"}</button>
                <button type="button" class="danger" data-action="delete">삭제</button>
              </div>
            </div>
            <div class="badges">
              <span class="badge">${escapeHtml(entry.knowledge_type)}</span>
              <span class="badge">${escapeHtml(entry.review_status)}</span>
              <span class="badge ${isNearHard ? "rule" : ""}">${escapeHtml(entry.enforcement_level || "soft")}</span>
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

    function setEditorEntry(entry) {
      selectedEntryId = entry.id;
      editorTitle.textContent = entry.title || entry.id;
      editorMeta.textContent = `${entry.id} · created ${entry.created_at || ""} · updated ${entry.updated_at || ""}`;
      editFields.raw_text.value = entry.raw_text || "";
      editFields.title.value = entry.title || "";
      editFields.knowledge_type.value = entry.knowledge_type || "soft_hint";
      editFields.review_status.value = entry.review_status || "draft";
      editFields.enforcement_level.value = entry.enforcement_level || "soft";
      editFields.priority.value = entry.priority ?? 50;
      editFields.confidence.value = entry.confidence ?? 0.5;
      editFields.active.value = entry.active ? "true" : "false";
      editFields.aliases.value = valueText(entry.aliases);
      editFields.match_fields.value = valueText(entry.match_fields);
      editFields.applies_when.value = entry.applies_when || "";
      editFields.hint.value = entry.hint || "";
      editFields.target_major_job.value = entry.target_major_job || "";
      editFields.target_sub_job.value = entry.target_sub_job || "";
      editFields.target_device.value = entry.target_device || "";
      editFields.target_unit_job.value = entry.target_unit_job || "";
      editFields.target_detail_job_1.value = entry.target_detail_job_1 || "";
      editFields.target_detail_job_2.value = entry.target_detail_job_2 || "";
      editFields.clear_conflicts.checked = false;
    }

    function renderUsage(usage) {
      if (!usage || !usage.usage_count) {
        usagePanel.innerHTML = "아직 분류 사용 기록이 없습니다.";
        return;
      }
      const needsRate = Math.round((usage.needs_review_rate || 0) * 1000) / 10;
      const recent = (usage.recent || []).slice(0, 8).map((item) => `
        <div class="usage-row">
          <div>${escapeHtml(item.final_major_job)} / ${escapeHtml(item.final_sub_job)} / ${escapeHtml(item.final_unit_job)}</div>
          <div>score ${escapeHtml(Number(item.match_score || 0).toFixed(2))} · needs_review ${item.needs_review ? "Y" : "N"}</div>
          <div>${escapeHtml(item.created_at)}</div>
        </div>
      `).join("");
      usagePanel.innerHTML = `
        <div>사용 ${escapeHtml(usage.usage_count)}회 · 분류 ${escapeHtml(usage.classification_count)}건 · 평균 score ${escapeHtml(Number(usage.avg_match_score || 0).toFixed(2))}</div>
        <div>needs_review ${escapeHtml(usage.needs_review_count)}건 (${escapeHtml(needsRate)}%)</div>
        ${recent}
      `;
    }

    function renderRevisions(revisions) {
      if (!revisions || !revisions.length) {
        revisionPanel.innerHTML = "변경 이력이 없습니다.";
        return;
      }
      revisionPanel.innerHTML = revisions.slice(0, 12).map((revision) => {
        const snapshot = revision.snapshot || {};
        return `
          <div class="revision-row">
            <div>${escapeHtml(revision.action)} · #${escapeHtml(revision.id)}</div>
            <div>${escapeHtml(revision.created_at)}</div>
            <div>${escapeHtml(snapshot.title || snapshot.raw_text || "")}</div>
            <button type="button" data-action="restore-revision" data-revision-id="${escapeHtml(revision.id)}">이 버전으로 복원</button>
          </div>
        `;
      }).join("");
    }

    async function openEditor(id) {
      const payload = await requestJson(`/api/knowledge/${id}`);
      setEditorEntry(payload.item);
      renderUsage(payload.usage);
      renderRevisions(payload.revisions);
      editorModal.hidden = false;
    }

    function editorPayload() {
      return {
        raw_text: editFields.raw_text.value.trim(),
        title: editFields.title.value.trim(),
        knowledge_type: editFields.knowledge_type.value,
        review_status: editFields.review_status.value,
        enforcement_level: editFields.enforcement_level.value,
        priority: Number(editFields.priority.value || 50),
        confidence: Number(editFields.confidence.value || 0.5),
        active: editFields.active.value === "true",
        aliases: splitValues(editFields.aliases.value),
        match_fields: splitValues(editFields.match_fields.value),
        applies_when: editFields.applies_when.value.trim(),
        hint: editFields.hint.value.trim(),
        target_major_job: editFields.target_major_job.value.trim(),
        target_sub_job: editFields.target_sub_job.value.trim(),
        target_device: editFields.target_device.value.trim(),
        target_unit_job: editFields.target_unit_job.value.trim(),
        target_detail_job_1: editFields.target_detail_job_1.value.trim(),
        target_detail_job_2: editFields.target_detail_job_2.value.trim(),
        clear_conflicts: editFields.clear_conflicts.checked
      };
    }

    async function saveEditor(event) {
      event.preventDefault();
      if (!selectedEntryId) return;
      const payload = editorPayload();
      if (!payload.raw_text) {
        setStatus("원문 입력 필요");
        editFields.raw_text.focus();
        return;
      }
      setStatus("편집 저장 중");
      try {
        const result = await requestJson(`/api/knowledge/${selectedEntryId}/edit`, {
          method: "POST",
          body: JSON.stringify(payload)
        });
        await loadEntries();
        await openEditor(result.item.id);
        setStatus("편집 저장됨");
      } catch (error) {
        setStatus(error.message);
      }
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

    async function importNdjsonFile() {
      const file = ndjsonFileEl.files && ndjsonFileEl.files[0];
      if (!file) {
        setStatus("NDJSON 파일 선택 필요");
        ndjsonFileEl.focus();
        return;
      }
      importNdjsonBtn.disabled = true;
      setStatus("NDJSON 읽는 중");
      try {
        const text = await file.text();
        const payload = await requestJson("/api/knowledge/import-ndjson", {
          method: "POST",
          body: JSON.stringify({ text })
        });
        ndjsonFileEl.value = "";
        setStatus(`${payload.created_count || 0}개 가져옴`);
        await loadEntries();
      } catch (error) {
        setStatus(error.message);
      } finally {
        importNdjsonBtn.disabled = false;
      }
    }

    saveBtn.addEventListener("click", saveEntry);
    importBtn.addEventListener("click", importTextFile);
    importNdjsonBtn.addEventListener("click", importNdjsonFile);
    exportBtn.addEventListener("click", () => {
      window.location.href = `/api/knowledge/export?scope=${encodeURIComponent(exportScope.value)}`;
    });
    [searchInput, statusFilter, activeFilter, enforcementFilter].forEach((element) => {
      element.addEventListener("input", () => renderEntries());
      element.addEventListener("change", () => renderEntries());
    });
    closeEditorBtn.addEventListener("click", () => {
      editorModal.hidden = true;
    });
    editorModal.addEventListener("click", (event) => {
      if (event.target === editorModal) {
        editorModal.hidden = true;
      }
    });
    editorForm.addEventListener("submit", saveEditor);
    refreshDetailBtn.addEventListener("click", () => {
      if (selectedEntryId) {
        openEditor(selectedEntryId).catch((error) => setStatus(error.message));
      }
    });
    revisionPanel.addEventListener("click", async (event) => {
      const button = event.target.closest("button[data-action='restore-revision']");
      if (!button || !selectedEntryId) return;
      const revisionId = Number(button.dataset.revisionId || 0);
      if (!revisionId || !confirm("선택한 revision으로 현재 지식을 복원할까요?")) return;
      try {
        const payload = await requestJson(`/api/knowledge/${selectedEntryId}/restore`, {
          method: "POST",
          body: JSON.stringify({ revision_id: revisionId })
        });
        await loadEntries();
        await openEditor(payload.item.id);
        setStatus("복원됨");
      } catch (error) {
        setStatus(error.message);
      }
    });
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
        if (action === "edit") {
          await openEditor(id);
          return;
        } else if (action === "toggle") {
          const active = item.classList.contains("inactive");
          await requestJson(`/api/knowledge/${id}/active`, {
            method: "POST",
            body: JSON.stringify({ active })
          });
        } else if (action === "approve") {
          await requestJson(`/api/knowledge/${id}/metadata`, {
            method: "POST",
            body: JSON.stringify({ knowledge_type: "verified_rule", review_status: "approved", enforcement_level: "strong" })
          });
        } else if (action === "draft") {
          await requestJson(`/api/knowledge/${id}/metadata`, {
            method: "POST",
            body: JSON.stringify({ knowledge_type: "soft_hint", review_status: "draft", enforcement_level: "soft" })
          });
        } else if (action === "near-hard-rule") {
          await requestJson(`/api/knowledge/${id}/metadata`, {
            method: "POST",
            body: JSON.stringify({ enforcement_level: "near_hard" })
          });
        } else if (action === "strong-rule") {
          await requestJson(`/api/knowledge/${id}/metadata`, {
            method: "POST",
            body: JSON.stringify({ enforcement_level: "strong" })
          });
        } else if (action === "clear-conflicts") {
          await requestJson(`/api/knowledge/${id}/metadata`, {
            method: "POST",
            body: JSON.stringify({ clear_conflicts: true })
          });
        } else if (action === "delete") {
          if (!confirm("이 지식을 삭제할까요?")) return;
          await requestJson(`/api/knowledge/${id}`, { method: "DELETE" });
          if (selectedEntryId === id) {
            editorModal.hidden = true;
            selectedEntryId = "";
          }
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


def draft_from_api_payload(payload: dict[str, Any]) -> KnowledgeDraft:
    return KnowledgeDraft(
        knowledge_type=payload.get("knowledge_type", "soft_hint"),
        title=payload.get("title", ""),
        aliases=payload.get("aliases", []),
        match_fields=payload.get("match_fields", []),
        applies_when=payload.get("applies_when", ""),
        hint=payload.get("hint", ""),
        target_major_job=payload.get("target_major_job", ""),
        target_sub_job=payload.get("target_sub_job", ""),
        target_device=payload.get("target_device", ""),
        target_unit_job=payload.get("target_unit_job", ""),
        target_detail_job_1=payload.get("target_detail_job_1", ""),
        target_detail_job_2=payload.get("target_detail_job_2", ""),
        priority=payload.get("priority", 50),
        confidence=payload.get("confidence", 0.5),
        validation_errors=payload.get("validation_errors", []),
    )


class KnowledgeRequestHandler(BaseHTTPRequestHandler):
    store: JobKnowledgeStore
    normalizer: KnowledgeNormalizer | None
    taxonomy: Taxonomy | None
    allow_fallback_normalizer: bool

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"", "/", "/index.html"}:
            self._send_html(INDEX_HTML)
            return
        if path == "/api/knowledge/export":
            self._handle_export(parsed.query)
            return
        if path == "/api/knowledge":
            items = [item.to_api_dict() for item in self.store.list_recent(limit=100)]
            self._send_json({"items": items})
            return
        if path.startswith("/api/knowledge/") and path.endswith("/revisions"):
            entry_id = path.removeprefix("/api/knowledge/").removesuffix("/revisions").strip("/")
            self._send_json({"items": self.store.list_revisions(entry_id)})
            return
        if path.startswith("/api/knowledge/") and path.endswith("/usage"):
            entry_id = path.removeprefix("/api/knowledge/").removesuffix("/usage").strip("/")
            self._send_json(self.store.usage_summary(entry_id))
            return
        if path.startswith("/api/knowledge/"):
            entry_id = path.removeprefix("/api/knowledge/").strip("/")
            entry = self.store.get(entry_id)
            if not entry:
                self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(
                {
                    "item": entry.to_api_dict(),
                    "revisions": self.store.list_revisions(entry_id),
                    "usage": self.store.usage_summary(entry_id),
                }
            )
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
        if path == "/api/knowledge/import-ndjson":
            self._handle_import_ndjson()
            return
        if path.startswith("/api/knowledge/") and path.endswith("/active"):
            entry_id = path.removeprefix("/api/knowledge/").removesuffix("/active").strip("/")
            self._handle_set_active(entry_id)
            return
        if path.startswith("/api/knowledge/") and path.endswith("/edit"):
            entry_id = path.removeprefix("/api/knowledge/").removesuffix("/edit").strip("/")
            self._handle_edit(entry_id)
            return
        if path.startswith("/api/knowledge/") and path.endswith("/metadata"):
            entry_id = path.removeprefix("/api/knowledge/").removesuffix("/metadata").strip("/")
            self._handle_update_metadata(entry_id)
            return
        if path.startswith("/api/knowledge/") and path.endswith("/restore"):
            entry_id = path.removeprefix("/api/knowledge/").removesuffix("/restore").strip("/")
            self._handle_restore(entry_id)
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

    def _handle_import_ndjson(self) -> None:
        try:
            payload = self._read_json(max_bytes=MAX_IMPORT_BODY_BYTES)
            raw_text = str(payload.get("text", "")).strip()
            if not raw_text:
                raise ValueError("NDJSON text is blank")
            imported = self.store.import_ndjson_text(raw_text, source="web_ndjson_import")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json(
            {
                "created_count": len(imported),
                "items": [item.to_api_dict() for item in imported],
            },
            status=HTTPStatus.CREATED,
        )

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

    def _handle_edit(self, entry_id: str) -> None:
        try:
            payload = self._read_json()
            draft = draft_from_api_payload(payload)
            if self.taxonomy:
                draft = validate_draft_against_taxonomy(draft, self.taxonomy)
            active = payload.get("active")
            entry = self.store.update_entry(
                entry_id,
                draft,
                raw_text=payload.get("raw_text"),
                active=active if isinstance(active, bool) else None,
                review_status=payload.get("review_status"),
                enforcement_level=payload.get("enforcement_level"),
                clear_conflicts=bool(payload.get("clear_conflicts", False)),
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

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
                enforcement_level=payload.get("enforcement_level"),
                clear_conflicts=bool(payload.get("clear_conflicts", False)),
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if not entry:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json({"item": entry.to_api_dict()})

    def _handle_restore(self, entry_id: str) -> None:
        try:
            payload = self._read_json()
            revision_id = int(payload.get("revision_id", 0))
            if revision_id <= 0:
                raise ValueError("revision_id is required")
            entry = self.store.restore_revision(entry_id, revision_id)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if not entry:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json({"item": entry.to_api_dict()})

    def _handle_export(self, query: str) -> None:
        review_scope = parse_qs(query).get("scope", ["approved"])[0]
        try:
            text = self.store.export_ndjson_text(review_scope=review_scope)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_text(
            text,
            content_type="application/x-ndjson; charset=utf-8",
            filename=f"job_knowledge_{review_scope}.ndjson",
        )

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

    def _send_text(
        self,
        text: str,
        *,
        content_type: str = "text/plain; charset=utf-8",
        filename: str | None = None,
    ) -> None:
        encoded = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
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
        normalizer = KnowledgeNormalizer(LLMSettings.from_env(role="knowledge"), taxonomy=taxonomy)
    except Exception as exc:
        if not args.allow_fallback_normalizer:
            print(f"LLM normalizer unavailable: {exc}")
            print(
                "Set KNOWLEDGE_INTERNAL_LLM_*, KNOWLEDGE_ALIBABA_*, or KNOWLEDGE_OPENAI_* env vars, "
                "or run with --allow-fallback-normalizer."
            )
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
