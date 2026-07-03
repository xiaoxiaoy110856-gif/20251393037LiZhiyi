function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function applyInlineFormatting(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return html;
}

function splitTableRow(line) {
  let trimmed = line.trim();
  if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
  if (trimmed.endsWith("|")) trimmed = trimmed.slice(0, -1);
  return trimmed.split("|").map((cell) => cell.trim());
}

function isTableSeparator(line) {
  const cells = splitTableRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function renderTable(block) {
  const lines = block
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2 || !lines.every((line) => line.includes("|")) || !isTableSeparator(lines[1])) {
    return null;
  }

  const headers = splitTableRow(lines[0]);
  const rows = lines.slice(2).map(splitTableRow);
  const headHtml = headers.map((cell) => `<th>${applyInlineFormatting(cell)}</th>`).join("");
  const bodyHtml = rows
    .map((row) => `<tr>${row.map((cell) => `<td>${applyInlineFormatting(cell)}</td>`).join("")}</tr>`)
    .join("");

  return `<div class="table-wrap"><table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`;
}

export function formatRichText(text) {
  const codeBlocks = [];
  let working = String(text || "");

  working = working.replace(/```([\s\S]*?)```/g, (_, code) => {
    const token = `__CODE_BLOCK_${codeBlocks.length}__`;
    codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    return token;
  });

  const blocks = working
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);

  let html = blocks
    .map((block) => {
      const tableHtml = renderTable(block);
      if (tableHtml) return tableHtml;

      if (/^[-*]\s+/m.test(block)) {
        const items = block
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line) => line.replace(/^[-*]\s+/, ""));
        return `<ul>${items.map((item) => `<li>${applyInlineFormatting(item)}</li>`).join("")}</ul>`;
      }

      if (/^\d+\.\s+/m.test(block)) {
        const items = block
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line) => line.replace(/^\d+\.\s+/, ""));
        return `<ol>${items.map((item) => `<li>${applyInlineFormatting(item)}</li>`).join("")}</ol>`;
      }

      return `<p>${applyInlineFormatting(block).replace(/\n/g, "<br>")}</p>`;
    })
    .join("");

  codeBlocks.forEach((block, index) => {
    html = html.replaceAll(`__CODE_BLOCK_${index}__`, block);
  });

  return html || `<p>${escapeHtml(text || "")}</p>`;
}

export async function typesetMath(root) {
  if (window.MathJax?.typesetPromise && root) {
    try {
      await window.MathJax.typesetPromise([root]);
    } catch (error) {
      console.warn("MathJax typeset failed:", error);
    }
  }
}
