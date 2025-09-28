from playwright.sync_api import sync_playwright
import time

def fetch_main_container_images_only(url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        time.sleep(3)  # 렌더링 대기

        # iframe 내부에 실제 블로그 본문이 있을 경우 대응
        frame = next((f for f in page.frames if "PostView" in f.url), page)

        # 본문 텍스트 추출
        try:
            content_text = frame.locator("div.se-main-container").inner_text()
        except:
            try:
                content_text = frame.locator("div#postViewArea").inner_text()
            except:
                content_text = "[본문 추출 실패]"

        # 이미지 추출: se-main-container 내부의 <img>만, 전처리 대상만
        image_urls = []
        try:
            container = frame.locator("div.se-main-container").first
            image_elements = container.locator("img").all()
            filtered = []

            for img in image_elements:
                src = img.get_attribute("src")
                if src and src.startswith("http"):
                    filtered.append(src)

            # ✅ 앞 2장 + 뒤 2장만 추출
            if len(filtered) <= 4:
                image_urls = filtered
            else:
                image_urls = filtered[:2] + filtered[-2:]

        except:
            pass

        browser.close()
        return {
            "url": url,
            "text": content_text,
            "images": image_urls  # 🔥 필터링된 4장만 반환
        }
