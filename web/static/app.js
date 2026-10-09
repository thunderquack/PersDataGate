const $ = (id) => document.getElementById(id);
let receivedText = "", restoredText = "", sentRequest = null;
let pipelineId = "";

function setText(id, text, empty = false) {
  const node = $(id);
  node.textContent = text;
  node.classList.toggle("empty", empty);
}

function showRequest(request) {
  sentRequest = request;
  setText("request", `Модель: ${request.model}\nИнструкции:\n${request.instructions}\n\nВвод:\n${request.input}\n\nСохранение: ${request.store}`);
}

function showRows(id, rows, columns, emptyMessage, unavailableMessage) {
  const container = $(id);
  container.replaceChildren();
  container.classList.toggle("empty", !Array.isArray(rows) || rows.length === 0);
  if (!Array.isArray(rows)) {
    container.textContent = unavailableMessage;
    return;
  }
  if (rows.length === 0) {
    container.textContent = emptyMessage;
    return;
  }
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  columns.forEach(([, label]) => {
    const cell = document.createElement("th");
    cell.textContent = label;
    head.append(cell);
  });
  const body = table.createTBody();
  rows.forEach((row) => {
    const tr = body.insertRow();
    columns.forEach(([key]) => {
      const cell = tr.insertCell();
      const value = row && row[key];
      cell.textContent = Array.isArray(value) ? value.join(", ") : value == null ? "—" : String(value);
    });
  });
  container.append(table);
}

function showSteps(result) {
  const steps = result.steps || {};
  const sources = {gliner: "GLiNER 2.5", rule: "Правило"};
  const statuses = {restored: "Восстановлено", unknown: "Неизвестный маркер"};
  const reasons = {
    saved_value: "Сохранённое значение",
  };
  showRows("entities", steps.entities?.map((row) => ({...row, source: sources[row.source] || row.source})), [
    ["value", "Найденный текст"], ["code", "Тип"], ["source", "Источник"], ["start", "Начало"], ["end", "Конец"],
  ], "Персональные данные не найдены.", "Сервер не передал сведения о распознанных данных.");
  showRows("persons", steps.persons, [
    ["marker", "Маркер"], ["canonical", "Связанное имя"], ["aliases", "Варианты упоминания"],
  ], "Упоминания людей не найдены.", "Сервер не передал результат связывания упоминаний.");
  if (typeof steps.anonymized_text === "string") setText("prepared", steps.anonymized_text, !steps.anonymized_text);
  else if (typeof result.prepared_text === "string") setText("prepared", result.prepared_text, !result.prepared_text);
  else setText("prepared", "Сервер не передал псевдонимизированный текст.", true);
  showRows("restorations", steps.restorations?.map((row) => ({...row,
    status: statuses[row.status] || row.status, reason: reasons[row.reason] || row.reason})), [
    ["marker", "Маркер"], ["value", "Подставленное значение"], ["status", "Результат"], ["reason", "Причина"], ["start", "Начало"], ["end", "Конец"],
  ], "В ответе нет маркеров для восстановления.", "Сервер не передал сведения о восстановленных значениях.");
}

function clearSteps() {
  showRows("entities", null, [], "", "Ожидание данных этапа.");
  showRows("persons", null, [], "", "Ожидание данных этапа.");
  showRows("restorations", null, [], "", "Ожидание данных этапа.");
  setText("prepared", "Ожидание данных этапа.", true);
}

$("process").addEventListener("click", async () => {
  const button = $("process"); button.disabled = true; receivedText = ""; restoredText = ""; sentRequest = null;
  pipelineId = crypto.randomUUID();
  $("status").className = ""; $("status").textContent = `Обработка… · GUID: ${pipelineId}`; $("warnings").textContent = "";
  clearSteps();
  setText("request", "Ожидание результата…", true); setText("received", "Ожидание ответа…", true); setText("restored", "Ожидание восстановления…", true);
  try {
    const response = await fetch("/api/process", {method:"POST", headers:{"Content-Type":"application/json", "X-Pipeline-ID":pipelineId}, body:JSON.stringify({document:$("document").value,prompt:$("prompt").value})});
    const result = await response.json();
    pipelineId = result.pipeline_id || response.headers.get("X-Pipeline-ID") || pipelineId;
    showSteps(result);
    if (result.sent_request) showRequest(result.sent_request);
    else if (result.prepared_text) setText("request", `Подготовленный псевдонимизированный текст:\n${result.prepared_text}`);
    else setText("request", "Запрос в OpenAI не сформирован.", true);
    if (typeof result.received_text === "string") { receivedText = result.received_text; setText("received", receivedText, !receivedText); }
    if (typeof result.restored_text === "string") { restoredText = result.restored_text; setText("restored", restoredText, !restoredText); }
    $("warnings").textContent = (result.warnings || []).join(" ");
    if (!response.ok) throw new Error(result.error?.message || "Не удалось обработать текст.");
    $("status").textContent = `Готово · GUID: ${pipelineId}`;
  } catch (error) {
    $("status").className = "error"; $("status").textContent = `${error.message || "Ошибка обработки."} · GUID: ${pipelineId}`;
    if (!sentRequest && !$("request").textContent.includes("Подготовленный")) setText("request", "Запрос в OpenAI не сформирован.", true);
    if (!receivedText) setText("received", "Ответ пока не получен.", true);
    if (!restoredText) setText("restored", "Ответ пока не восстановлен.", true);
  } finally { button.disabled = false; }
});

document.querySelectorAll("[data-copy]").forEach((button) => button.addEventListener("click", async () => {
  const text = button.dataset.copy === "request" ? (sentRequest
    ? `Модель: ${sentRequest.model}\nИнструкции:\n${sentRequest.instructions}\n\nВвод:\n${sentRequest.input}\n\nСохранение: ${sentRequest.store}`
    : $("request").textContent)
    : button.dataset.copy === "received" ? receivedText : restoredText;
  if (text) {
    try { await navigator.clipboard.writeText(text); const old = button.textContent; button.textContent = "Скопировано"; setTimeout(() => button.textContent = old, 1200); }
    catch (_) { $("status").className = "error"; $("status").textContent = `Не удалось скопировать: браузер запретил доступ к буферу обмена.${pipelineId ? ` · GUID: ${pipelineId}` : ""}`; }
  }
}));
