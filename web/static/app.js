const $ = (id) => document.getElementById(id);
let receivedText = "", restoredText = "", sentRequest = null;
function setText(id, text, empty = false) { const node = $(id); node.textContent = text; node.classList.toggle("empty", empty); }

$("process").addEventListener("click", async () => {
  const button = $("process"); button.disabled = true; receivedText = ""; restoredText = ""; sentRequest = null;
  $("status").className = ""; $("status").textContent = "Обработка…"; $("warnings").textContent = "";
  setText("request", "Обработка…"); setText("received", "Ожидание ответа…"); setText("restored", "Ожидание ответа…");
  try {
    const response = await fetch("/api/process", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({document:$("document").value,prompt:$("prompt").value})});
    const result = await response.json();
    if (result.sent_request) {
      sentRequest = result.sent_request;
      setText("request", `Instructions:\n${sentRequest.instructions}\n\nInput:\n${sentRequest.input}\n\nStore: ${sentRequest.store}`);
    } else if (result.prepared_text) setText("request", `Prepared pseudonymized input (OpenAI request failed):\n${result.prepared_text}`);
    if (result.received_text) { receivedText = result.received_text; setText("received", receivedText); }
    if (result.restored_text !== undefined) { restoredText = result.restored_text; setText("restored", restoredText); }
    if (!response.ok) throw new Error(result.error?.message || "Не удалось обработать текст.");
    $("warnings").textContent = (result.warnings || []).join(" "); $("status").textContent = "Готово";
  } catch (error) {
    $("status").className = "error"; $("status").textContent = error.message || "Ошибка обработки.";
    if (!receivedText) setText("received", "Ответ пока не получен.", true);
    if (!restoredText) setText("restored", "Ответ пока не восстановлен.", true);
  } finally { button.disabled = false; }
});

document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab === button));
  $("received").classList.toggle("hidden", button.dataset.tab !== "received");
  $("restored").classList.toggle("hidden", button.dataset.tab !== "restored");
}));
document.querySelectorAll("[data-copy]").forEach((button) => button.addEventListener("click", async () => {
  const text = button.dataset.copy === "request"
    ? (sentRequest ? `Instructions:\n${sentRequest.instructions}\n\nInput:\n${sentRequest.input}\n\nStore: ${sentRequest.store}` : "")
    : (document.querySelector("#restored").classList.contains("hidden") ? receivedText : restoredText);
  if (text) { await navigator.clipboard.writeText(text); const old = button.textContent; button.textContent = "Скопировано"; setTimeout(() => button.textContent = old, 1200); }
}));
