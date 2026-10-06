const pages = [
  ["index", "开始这里", "首页与学习路线"],
  ["01-foundations", "01", "零基础概念"],
  ["02-project-map", "02", "项目地图与架构"],
  ["03-model-loading", "03", "配置与权重加载"],
  ["04-qwen-forward", "04", "Qwen 前向计算"],
  ["05-kv-cache", "05", "KV Cache"],
  ["06-paged-kv", "06", "Paged KV Cache"],
  ["07-paged-attention", "07", "CUDA Paged Attention"],
  ["08-scheduler", "08", "Scheduler 调度器"],
  ["09-engine", "09", "Continuous Engine"],
  ["10-testing", "10", "测试、指标与 Benchmark"],
  ["11-reading-lab", "11", "源码阅读实战"],
  ["glossary", "附录", "术语表与速查"]
];

const current = document.body.dataset.page || "index";
const sidebar = document.createElement("aside");
sidebar.className = "sidebar";
sidebar.id = "tutorial-sidebar";
const links = pages.map(([id, number, title]) =>
  `<a class="side-link ${id === current ? "active" : ""}" href="${id === "index" ? "index.html" : id + ".html"}"><span class="side-num">${number}</span><span>${title}</span></a>`
).join("");
sidebar.innerHTML = `
  <a class="brand" href="index.html">mini·<span>vLLM</span> 教程</a>
  <span class="version">v1.1.0 · 源码教程</span>
  <nav class="nav-group"><div class="nav-label">从零到源码</div>${links}</nav>
  <div class="side-footer">所有页面均可直接用 <code>file://</code> 打开；源码链接相对当前仓库解析。</div>`;

const layout = document.querySelector(".layout");
if (layout) layout.prepend(sidebar);

const mobile = document.createElement("div");
mobile.className = "mobile-bar";
mobile.innerHTML = `<button class="menu-button" aria-label="打开目录">☰ 目录</button><span class="mobile-title">mini-vLLM 从零教程</span>`;
document.body.prepend(mobile);
mobile.querySelector("button").addEventListener("click", () => sidebar.classList.toggle("open"));
document.addEventListener("click", (event) => {
  if (window.innerWidth <= 920 && sidebar.classList.contains("open") && !sidebar.contains(event.target) && !mobile.contains(event.target)) sidebar.classList.remove("open");
});

document.querySelectorAll("pre code").forEach((code) => {
  code.innerHTML = code.innerHTML
    .replace(/(^|\n)(\s*#.*)/g, '$1<span class="comment">$2</span>')
    .replace(/\b(begin_append|commit_append|abort_append|prefill|decode_batch|schedule_step|apply_step_results)\b/g, '<span class="hot">$1</span>');
});
