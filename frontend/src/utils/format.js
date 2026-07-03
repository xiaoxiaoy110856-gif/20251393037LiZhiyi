function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

const RL_ARROW = String.raw`(?:→|->|⇒|=>|-->)`;
const RL_TRANSITION_FORMULA = String.raw`\[s_t \xrightarrow{\;\pi_\theta\;} a_t \xrightarrow{\;\mathcal{E}\;} (r_{t+1}, s_{t+1})\]`;
const RL_TRANSITION_PATTERN = String.raw`状态\s*s(?:_|\\_)?(?:\{?t\}?)\s*${RL_ARROW}\s*策略\s*(?:π|pi|\\pi|π_\s*θ|\\pi_\s*\\theta|\\pi_\{?\\theta\}?)?\s*${RL_ARROW}\s*动作\s*a(?:_|\\_)?(?:\{?t\}?)\s*${RL_ARROW}\s*环境\s*${RL_ARROW}\s*奖励\s*r(?:_|\\_)?(?:\{?t\s*\+?\s*1\}?|t\s*\+?\s*1)\s*(?:\+|和|、)\s*新状态\s*s(?:_|\\_)?(?:\{?t\s*\+?\s*1\}?|t\s*\+?\s*1)`;

function stripCodeLanguage(code) {
  const trimmed = String(code || "").trim();
  const lines = trimmed.split(/\r?\n/);
  if (lines.length > 1 && /^[a-zA-Z0-9_-]{1,24}$/.test(lines[0].trim())) {
    return lines.slice(1).join("\n").trim();
  }
  return trimmed;
}

function isRlTransitionChain(text) {
  return new RegExp(`^\\s*${RL_TRANSITION_PATTERN}\\s*$`, "i").test(String(text || "").replace(/\s+/g, " ").trim());
}

function replaceRlTransitionChains(text) {
  return String(text || "").replace(new RegExp(RL_TRANSITION_PATTERN, "gi"), `\n\n${RL_TRANSITION_FORMULA}\n\n`);
}

function applyInlineFormatting(text) {
  let html = escapeHtml(text);
  html = html.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, '<img class="chat-generated-image" src="$2" alt="$1" />');
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
    const codeBody = stripCodeLanguage(code);
    if (isRlTransitionChain(codeBody)) {
      codeBlocks.push(`<div class="math-block">${RL_TRANSITION_FORMULA}</div>`);
    } else {
      codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    }
    return token;
  });
  working = replaceRlTransitionChains(working);

  const blocks = working
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);

  let html = blocks
    .map((block) => {
      if (/^__CODE_BLOCK_\d+__$/.test(block)) return block;

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
  if (!window.MathJax && !document.getElementById("mathjax-script")) {
    const config = document.createElement("script");
    config.text = `
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
        },
        options: {
          skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
        }
      };
    `;
    document.head.appendChild(config);
    const script = document.createElement("script");
    script.id = "mathjax-script";
    script.async = true;
    script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js";
    document.head.appendChild(script);
    await new Promise((resolve) => {
      script.onload = resolve;
      script.onerror = resolve;
    });
  }

  if (window.MathJax?.typesetPromise && root) {
    try {
      await window.MathJax.typesetPromise([root]);
    } catch (error) {
      console.warn("MathJax typeset failed:", error);
    }
  }
}
