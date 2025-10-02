console.log("✅ content.js injected and running");

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "get_first_blog_url") {
    const anchor = document.querySelector("a.Vyg_WCkzlKBSk8usfmMA.OwwmICzrKXneAIOVrlrA");
    console.log("🔍 찾은 링크:", anchor?.href);

    if (anchor && anchor.href.includes("blog.naver.com")) {
      sendResponse({ url: anchor.href });
    } else {
      sendResponse({ error: "블로그 링크를 찾을 수 없습니다." });
    }
  }
});

function isRealBlogAnchor(anchor) {
  const href = anchor.href;
  const isBlog = href.includes("blog.naver.com");
  const isAdUrl = href.includes("adcr.") || href.includes("ader.") || href.includes("click.") || href.includes("brand/");
  const isAdDom = anchor.closest(".ad_section, .power_ad, .type_ad, .ad_area, .link_ad") || anchor.closest('[data-cr-area="ad"]');
  return isBlog && !isAdUrl && !isAdDom;
}

setTimeout(async function() {
    const blogAnchors = Array.from(document.querySelectorAll("a.Vyg_WCkzlKBSk8usfmMA.OwwmICzrKXneAIOVrlrA")).filter(isRealBlogAnchor);
    console.log(blogAnchors);
    console.log("🔗 분석 대상 링크 수:", blogAnchors.length);

    for (const anchor of blogAnchors) {
      const url = anchor.href;
      const parentCard = anchor.closest(".sds-comps-vertical-layout.sds-comps-full-layout.JTS3NufK1HH_Obhpw1_U");
      if (!url || !parentCard) continue;
      if (parentCard.querySelector(".ad-result-tag")) continue;

      const resultBox = document.createElement("div");
      resultBox.className = "ad-result-tag";
      resultBox.style.cssText = "margin-top:5px; padding:6px 10px; border-radius:6px; font-size:13px; background:#f5f5f5;";
      resultBox.textContent = "⏳ 광고 여부 분석 중...";
      parentCard.appendChild(resultBox);
      console.log("박스 추가 성공");
      console.log(parentCard);

      try {
        const res = await fetch("http://localhost:5000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url })
        });

        const data = await res.json();

        const label = data.label === 1 ? "📢 광고" : "✅ 비광고";
        const prob = (data.prob * 100).toFixed(1);
        const anomaly = data.anomaly_detected
          ? "⚠️ 이상 탐지됨"
          : data.anomaly_score === null
            ? "🟡 이상탐지 대상 아님"
            : "✅ 정상";

        const detailHtml = `
          <div class="ad-detail-box" style="margin-top:6px; display:none; background:#fff; border:1px solid #ddd; padding:10px; border-radius:6px; font-size:12px;">
            <b>🧾 OCR 결과:</b><br>
            ${
              (data.ocr_texts || []).map((t, i) => {
                const cleaned = t.trim();
                return cleaned.length > 0
                  ? `(${i + 1}) ${cleaned}`
                  : `(${i + 1}) ❌ 탐지되지 않음`;
              }).join("<br>")
            }
            <br><br><b>📸 사용된 이미지:</b><br>
            ${
              (data.imgUrls || []).map(url =>
                `<img src="${url}" style="max-width:100%; max-height:100px; margin:5px 0;">`
              ).join("<br>")
            }
            <br><br><b>🖼 전처리 이미지:</b><br>
            ${
              (data.ocr_images || []).map((b64, i) =>
                b64 && b64.length > 30
                  ? `<img src="data:image/png;base64,${b64}" style="max-width:100%; max-height:100px; margin:5px 0;">`
                  : `<i>(${i + 1}) ❌ 이미지 없음</i>`
              ).join("<br>")
            }
          </div>
        `;

        const detailBox = document.createElement("div");
        detailBox.innerHTML = detailHtml;

        const toggleBtn = document.createElement("button");
        toggleBtn.textContent = "상세 보기";
        toggleBtn.style.cssText = "margin-left: 10px; font-size: 12px; padding: 2px 6px; cursor: pointer;";

        toggleBtn.onclick = () => {
          const box = detailBox.querySelector(".ad-detail-box");
          box.style.display = box.style.display === "none" ? "block" : "none";
        };

        resultBox.innerHTML = `<b>${label}</b><br>확률: ${prob}%<br>${anomaly}`;
        resultBox.appendChild(toggleBtn);
        resultBox.appendChild(detailBox);

      } catch (err) {
        resultBox.textContent = "❌ 분석 실패";
        console.error("분석 오류:", err);
      }
    }
}, 3000);
