let cachedText = "";
let cachedImgCount = 0;

function extractBlogTextAndImages() {
  let main = document.querySelector(".se-main-container");
  if (main) {
    const text = Array.from(main.querySelectorAll("p")).map(p => p.innerText).join("\n");
    const imgCount = main.querySelectorAll("img").length;
    return { text, imgCount };
  }

  main = document.querySelector("#postViewArea");
  if (main) {
    const text = main.innerText.trim();
    const imgCount = main.querySelectorAll("img").length;
    return { text, imgCount };
  }

  const postArea = document.querySelectorAll(".se_component_wrap.sect_dsc");
  if (postArea.length > 0) {
    const text = Array.from(postArea).map(el => el.innerText.trim()).join("\n");
    const imgCount = Array.from(postArea).reduce((sum, el) => sum + el.querySelectorAll("img").length, 0);
    return { text, imgCount };
  }

  return { text: "❌ 본문을 찾을 수 없습니다.", imgCount: 0 };
}

chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
  chrome.scripting.executeScript(
    {
      target: { tabId: tabs[0].id, allFrames: true },
      func: extractBlogTextAndImages
    },
    (results) => {
      const infoDiv = document.getElementById("info");
      const valid = results.find(r => r.result && !r.result.text.startsWith("❌"));
      if (valid) {
        const { text, imgCount } = valid.result;
        cachedText = text;
        cachedImgCount = imgCount;
        infoDiv.textContent = `📷 이미지 개수: ${imgCount}`;
      } else {
        infoDiv.textContent = "❌ 본문을 찾을 수 없습니다.";
      }
    }
  );
});

document.getElementById("ad-0").addEventListener("click", () => sendLabel(0));
document.getElementById("ad-1").addEventListener("click", () => sendLabel(1));

function sendLabel(label) {
  fetch("http://localhost:5000/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content: cachedText,
      images: cachedImgCount,
      is_ad: label
    })
  })
    .then(res => res.json())
    .then(data => {
      alert("✅ 저장 완료: " + (label === 1 ? "광고" : "비광고"));
    })
    .catch(err => {
      alert("❌ 저장 실패");
      console.error(err);
    });
}
