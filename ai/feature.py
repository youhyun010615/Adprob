import pandas as pd
import re
from collections import Counter
from konlpy.tag import Okt
import numpy as np

okt = Okt()


sponsored_keywords = [
    "협찬", "제공", "광고", "체험단", "무료 제공", "서포터즈", "후원", "스폰서", "브랜드 지원","대가를 받고",
    "홍보", "제품 제공", "지원받음", "지원받은", "무상 제공", "무상으로 제공", "제공받은", "PPL", "지원", "제품 서포트",
    "브랜드 체험", "브랜드 측에서", "원고료", "광고성", "업체 제공", "광고 포함", "광고 문구", "광고 후기", "소정", "소정의",
    "제품을 제공받아", "브랜드로부터", "광고주", "광고성 글", "지원받아 작성", "협찬받아 작성", "제공받아 작성",
    "이 포스팅은", "이 글은", "해당 브랜드로부터", "제품을 무상으로", "업체로부터", "서포트를 받음", "광고 콘텐츠",
    "유료 광고 포함", "유료 협찬 포함", "본 제품은", "광고 목적", "PR 글", "광고 협찬", "제품 협찬", "브랜드 협찬",
    "서포터즈 활동", "제품을 받아보고", "스폰서십", "브랜드에서 보내준", "브랜드측에서 제공", "제품 리뷰 요청을 받아",
    "제작 지원", "홍보용으로 제공", "마케팅 목적", "광고용", "광고 리뷰", "광고 게시", "광고 안내", "광고 문구 포함", "무료제공", "브랜드지원", "대가를받고", "제품제공", "무상제공", "무상으로제공", "제품서포트",
    "브랜드체험", "브랜드측에서", "업체제공", "광고포함", "광고문구", "광고후기",
    "제품을제공받아", "광고성글", "지원받아작성", "협찬받아작성", "제공받아작성",
    "이포스팅은", "이글은", "해당브랜드로부터", "제품을무상으로", "업체로부터", "서포트를받음",
    "유료광고포함", "유료협찬포함", "본제품은", "광고목적", "PR글", "광고협찬", "제품협찬", "브랜드협찬",
    "서포터즈활동", "제품을받아보고", "브랜드에서보내준", "브랜드측에서제공", "제품리뷰요청을받아",
    "제작지원", "홍보용으로제공", "마케팅목적", "광고리뷰", "광고게시", "광고안내", "광고문구포함"
]

positive_keywords = [
    "좋다", "좋음", "좋아요", "좋습니다", "좋네요", "너무 좋네요", "참 좋네요", "괜찮다", "괜찮네요",
    "맘에 들어요", "마음에 들어요", "대만족", "만족해요", "만족합니다", "아주 만족", "정말 만족",
    "추천", "추천합니다", "추천드려요", "적극 추천", "강력 추천", "완전 추천", "강추", "강추합니다",
    "극추천", "진심 추천", "지인 추천", "재구매 의사", "재구매 예정", "또 살래요", "한 번 더 구매",
    "다시 사고 싶어요", "또 사고 싶어요", "진짜 좋아요", "정말 좋아요", "완전 좋아요", "엄청 좋아요", "너무 좋아요",
    "최고예요", "최고입니다", "최고다", "최애템", "인생템", "인생 템", "찐템", "찐템이에요", "역대급", "역대급이에요", "갓템",
    "대박템", "대박이에요", "완전 대박", "미친 제품", "미쳤어요", "미친 퀄리티", "신의 한 수", "신세계",
    "완전 맘에 들어요", "기대 이상", "기대한 것 이상", "사길 잘했어요", "후회 없어요", "진짜 만족해요",
    "없으면 안 돼요", "필수템", "없으면 허전해요", "하나쯤은 있어야 해요", "고민하다 샀는데", "고민할 필요 없어요",
    "사세요", "꼭 사세요", "필구템", "있으니까 너무 좋아요", "매우 만족", "만족도 최고", "가성비 최고",
    "퀄리티 미쳤어요", "구매 강추", "왜 이제 샀을까", "신세계에요", "완전 사랑해요", "완벽해요", "찢었어요",
    "기대 이상이에요", "생각보다 훨씬 좋아요", "정말 괜찮네요", "진짜 괜찮아요", "대박 만족", "역시 믿고 쓰는",
    "믿고 사는", "믿고 구매", "믿고 쓰는", "믿고 사용하는", "이번에도 만족", "한 번 써보세요", "두고두고 쓸 것 같아요",
    "기능 완벽", "디자인 예쁨", "디자인 너무 예뻐요", "고급스러움", "고급져요", "고급져", "깔끔한 디자인",
    "감성템", "감성 자극", "소장 가치 있음", "보는 순간 반함", "갖고 싶던 아이템", "갖고 싶었어요", "고민할 필요 없음",
    "진작 살걸", "안 샀으면 후회했을 듯", "가성비 갑", "가성비 킹", "가성비 미쳤어요", "가성비 만족", "이 가격에 이 퀄리티?",
    "돈 아깝지 않음", "돈 값함", "돈이 아깝지 않아요", "가격 대비 훌륭", "성능 최고", "성능 미쳤어요", "성능 만족",
    "가벼워서 좋아요", "편해서 좋아요", "착용감 좋아요", "부드러워요", "촉감 좋아요", "향기 좋아요", "향기 미쳤어요",
    "디테일 살아있음", "섬세하게 잘 만든", "정성 가득", "세심한 구성", "센스 있다", "센스 돋보임", "센스 최고",
    "아이디어 좋다", "이런 제품 처음 봐요", "새롭고 좋아요", "지금까지 중 최고", "역대급 만족", "인생템 발견",
    "하루종일 만족", "매일 쓰고 있어요", "매일 찾게 돼요", "없으면 허전함", "생활 필수템", "실용성 최고", "기능성 굿",
    "간편해서 좋아요", "정말 편리", "간편하게 사용", "활용도 높음", "활용도 최고", "여러모로 유용해요", "여러 개 사야 해요",
    "가족들도 만족", "부모님이 좋아하심", "친구도 사달라고 함", "주변에 추천함", "지인들에게 추천함", "주변 반응 좋음",
    "주변에서 예쁘대요", "주변에서 물어봄", "지인들에게 선물함", "선물용으로도 최고", "선물했더니 좋아했어요",
    "포장도 예쁨", "포장도 고급짐", "배송 빠름", "빠른 배송에 만족", "정확한 배송", "정품 인증", "믿을 수 있는 제품",
    "브랜드 믿고 구매", "AS 만족", "고객응대 좋음", "친절한 서비스", "재구매 100%", "또 사고 싶음", "없으면 안 될 물건",
    "요즘 가장 잘 쓰는", "이번 달 베스트", "평생템", "계속 쓰게 될 것 같아요", "리필 꼭 사야 함", "필수 구매 아이템"
]

price_keywords = [
    "가격", "가격대", "가격표", "정가", "할인가", "특가", "세일", "할인", "이벤트", "프로모션",
    "저렴", "저가", "고가", "비쌈", "비싸", "가성비", "합리적", "가성비 좋음", "저렴한", "저렴하게",
    "가성비 있는", "가격 대비", "가치 있는 가격", "가격 만족", "가격 만족도", "착한 가격", "혜자", "혜자템",
    "적당한 가격", "가격 괜찮음", "가격 착해요", "가격이 착하다", "비용 부담 없음", "부담 없는 가격", "가격 경쟁력",
    "만원", "만 원", "원", "원대", "천원", "천 원", "1만 원", "1만원", "10,000원", "10000원",
    "5천원", "오천원", "오천 원", "1만원대", "2만원대", "3만원대", "4만원대", "5만원대", "10만원대",
    "몇 천원", "몇 만원", "1~2만원", "2~3만원", "3~4만원", "1만원 이하", "1만원 미만", "5천원 이하", "무료",
    "가성비 미쳤다", "혜자스러운", "정신 못 차리는 가격", "말도 안 되는 가격", "찐으로 싸다", "찐 혜자", "가격 미침",
    "너무 싸다", "그 가격에 이 정도?", "돈값함", "이게 이 가격?", "이 가격 실화?", "이 가격이면 무조건 사야", "가격 미쳤어요",
    "이 정도 퀄리티에 이 가격?", "가격 듣고 놀람", "가격 보고 깜짝", "거의 공짜", "거저 줌", "미친 가성비", "혜자 가격", "가격 실화냐"
]

shopping_mall_domains = ["smartstore.naver.com", "musinsa.com", "coupang.com", "gmarket.co.kr", "11st.co.kr", "link.coupang", "bit.ly", "brand.naver.com"]

"""
negation_patterns = [
    r"(광고|협찬|스폰서|지원|체험단)\s*(?:을|를)?\s*(?:전혀|일절|일체)?\s*(받지\s*않|안\s*받|않았|받은\s*적\s*없|없어요|없습니다|아닙니다|아니에요|안\s*했|없었|아님|아녀)",
    r"(내\s*돈\s*내\s*산|내돈내산|본인\s*구매|자비로\s*구매|직접\s*구매|내\s*돈\s*주고\s*샀|내\s*돈\s*으로\s*직접\s*구매|사비로\s*구매|직접\s*결제|자체\s*구매|내\s*돈\s*지불)",
    r"(no|not|without|zero)\s*(sponsor|sponsorship|ad|advertisement|promotion|support)",
    r"(무\s*광고|비\s*광고|비\s*협찬|무\s*협찬|언\s*스폰서|노\s*스폰서|비\s*스폰서|노\s*광고|노\s*협찬)",
    r"(광고\s*아니에요|협찬\s*아닙니다|이건\s*광고\s*아님|이건\s*협찬\s*아님|스폰서\s*아닙니다|협찬\s*없이\s*구매)",
    r"(정말\s*내돈내산|100%\s*내돈내산|내\s*돈\s*주고\s*구매|광고\s*절대\s*아님|진짜\s*내돈내산)"
]
"""

# --- 부정 키워드 (감성 분석용) ---
negative_keywords = [
    "싫다", "별로", "실망", "나쁘다", "최악", "후회", "불만", "실망스럽", "비추천", "구리다",
    "별로다", "형편없다", "짜증", "짜증나다", "최악이다", "엉망", "불편하다", "이상하다",
    "불쾌", "불쾌하다", "불친절", "거슬린다", "별로였어요", "구려요", "별로에요",
    "실망이에요", "불만족", "별 한 개", "다신 안", "추천 안", "다시는", "구매하지 않을",
    "다신 안사요", "돈 아깝다", "돈 버렸다", "돈 낭비", "비추", "후회합니다",
    "다신 안 씀", "망했어요", "헛돈", "진절머리", "실패", "재구매 안 함",
    "아쉬워요", "찝찝", "찝찝해요", "의심스러움", "믿을 수 없음"
    "애매하다", "그냥 그래요", "별 차이 없음", "기대 이하", "다른 게 더", "과대광고",
    "생각보다 별로", "효과 없음", "느낌 없음", "불신", "믿기 힘듦", "이상한 냄새",
    "쓸데없음", "돈값 못함", "믿고 샀는데", "좋다길래 샀는데", "광고에 속음",
    "차라리 안 샀으면", "다시 안 삼", "기대했는데", "손해 본 느낌", "더 싼 게 나음",
    "유명한 거 치고", "광고만 번지르르", "좋다는 말만 무성", "추천하기 어려움"
]


# --- 부정 키워드 확인 함수 ---
"""
주어진 키워드가 부정 표현으로 사용되었는지 여부를 판단.
예: "광고를 받지 않았어요"는 광고 키워드가 포함되었지만 실제로는 비광고.
"""
def is_negated_in_sentence(text, keyword):
    neg_markers = r"(받지\s*않|안\s*받|않았|받은\s*적\s*없|없어요|없습니다|아닙니다|아니에요|안\s*했|없었|아님|아녀)"
    pattern = re.compile(
        rf"({neg_markers}.*?\b{re.escape(keyword)}\b|\b{re.escape(keyword)}\b.*?{neg_markers})"
    )
    return bool(re.search(pattern, text))





# --- 쇼핑몰 링크 포함 여부 확인 ---
"""
텍스트 내 사전 정의된 쇼핑몰 도메인 중 하나라도 포함되어 있는지 여부 반환.
"""
def has_shopping_mall_link(text):
    text = str(text)  # float이 들어와도 문자열로 강제 변환
    return int(any(domain in text for domain in shopping_mall_domains))

# --- 형태소 기반 키워드 등장 횟수 카운트 ---
def count_keywords_morph(text, keywords, morph_cache=None):
    text = str(text)
    morphs = morph_cache if morph_cache else okt.morphs(text)
    count = 0
    sentences = re.split(r"[.!?。!?]\s*", text)

    for kw in keywords:
        kw_morphs = okt.morphs(kw)
        kw_len = len(kw_morphs)

        for sentence in sentences:
            if kw not in sentence:
                continue
            if is_negated_in_sentence(sentence, kw):
                continue
            sentence_morphs = okt.morphs(sentence)
            for i in range(len(sentence_morphs) - kw_len + 1):
                if sentence_morphs[i:i + kw_len] == kw_morphs:
                    count += 1
    return count



# --- 외부 링크 수 카운트 ---
"""
텍스트 내 포함된 외부 URL 링크 개수 반환.
"""
def count_external_links(text):
    text = str(text)
    url_pattern = r'((https?:\/\/)?(www\.)?[\w\-]+\.(com|co\.kr|kr|net|org|shop|me|info|biz|tv|store)(\/\S*)?)'
    return len(re.findall(url_pattern, text))






# --- 최빈 단어 비율 계산 ---
"""
가장 많이 등장한 단어가 전체 텍스트에서 차지하는 비율 계산.
"""
def count_most_common_word_freq_ratio(text):
    stopwords = {"이", "그", "저", "것", "수", "등", "더", "또", "좀", "듯", "게",
                 "전", "후", "까지", "만", "에서", "으로", "하고", "에게", "보다",
                 "너무", "진짜", "정말", "아주", "매우", "해서", "해서요"}
    words = [w for w in str(text).split() if w not in stopwords and len(w) > 1]
    if not words:
        return 0
    word_counts = Counter(words)
    most_common = word_counts.most_common(1)[0][1]
    return most_common / len(words)




# --- 1인칭 대명사 비율 ---
def first_person_ratio(text):
    first_person_pronouns = ["나", "내", "저", "저는", "제가", "저도", "내가"]
    words = okt.morphs(str(text))
    if not words:
        return 0
    count = sum(w in first_person_pronouns for w in words)
    return count / len(words)




# --- !? + ~^ 비율 합산 ---
def exclamation_special_score(text):
    t = str(text)
    if not t:
        return 0.0
    length = len(t)
    return ((t.count("!") + t.count("?")) + len(re.findall(r"[~^]", t))) / length



# --- 어휘 다양성 --- (Type-Token Ratio)
"""
텍스트의 type-token ratio를 계산하여 어휘 다양성을 측정.
"""
def lexical_diversity_ratio(text):
    tokens = okt.morphs(str(text))
    if not tokens:
        return 0
    return len(set(tokens)) / len(tokens)






# --- 품사 분포 --- (명사 비율 반환)
def noun_ratio(text):
    tokens = okt.pos(str(text))
    total = len(tokens)
    if total == 0:
        return 0.0
    noun_count = sum(1 for word, tag in tokens if tag == "Noun")
    return noun_count / total

# --- 감성 점수 계산 ---
"""
긍정 키워드 - 부정 키워드 수를 기반으로 감성 점수 계산.
"""
def sentiment_score(text):
    morph_cache = okt.morphs(str(text))
    if not morph_cache:
        return 0
    pos_count = count_keywords_morph(text, positive_keywords, morph_cache=morph_cache)
    neg_count = count_keywords_morph(text, negative_keywords, morph_cache=morph_cache)
    return (pos_count - neg_count) / len(morph_cache)

# --- 광고 키워드가 초반에 등장하는지 비율 계산 ---
"""
전체 광고 키워드 중 초반 30% 이내 등장한 키워드 비율 계산.
"""
def sponsored_keyword_position_ratio(text):
    tokens = okt.morphs(str(text))
    if not tokens:
        return 0.0
    split_index = max(1, int(0.3 * len(tokens)))
    first_tokens = tokens[:split_index]

    def count_keywords_in_tokens(token_list, keywords):
        count = 0
        for kw in keywords:
            kw_morphs = okt.morphs(kw)
            kw_len = len(kw_morphs)
            for i in range(len(token_list) - kw_len + 1):
                if token_list[i:i + kw_len] == kw_morphs:
                    count += 1
        return count

    full_count = count_keywords_in_tokens(tokens, sponsored_keywords)
    if full_count == 0:
        return 0.0

    first_count = count_keywords_in_tokens(first_tokens, sponsored_keywords)
    return first_count / full_count



# 텍스트 전체 길이
def text_length(text):
    return len(str(text))

def text_length_log(text):
    return np.log1p(len(str(text)))  # log(1 + x) 형태로 음수 방지




# 이미지 수
def image_density_log(row):
    text_len = len(str(row["text"]))
    return row["image_count"] / np.log1p(text_len) if text_len > 0 else 0


def z_score_binarize(series, threshold=1.0):  # threshold는 필요시 조정
    mean = series.mean()
    std = series.std(ddof=0)
    z_scores = (series - mean) / (std if std > 0 else 1)  # 분모 0 방지
    return (z_scores > threshold).astype(int)


# --- 분석 함수 ---
def analyze_reviews(df):
    df["sponsored_keywords_count"] = df["text"].apply(lambda x: count_keywords_morph(x, sponsored_keywords))
    df["positive_bias_keywords"] = df["text"].apply(lambda x: count_keywords_morph(x, positive_keywords))
    df["external_links_count"] = df["text"].apply(count_external_links)
    df["text_length_log"] = df["text"].apply(text_length_log)
    df["most_common_word_freq_ratio"] = df["text"].apply(count_most_common_word_freq_ratio)
    df["price_mentions"] = df["text"].apply(lambda x: count_keywords_morph(x, price_keywords))
    df["first_person_ratio"] = df["text"].apply(first_person_ratio).clip(upper=0.15)
    
    df["lexical_diversity_ratio"] = df["text"].apply(lexical_diversity_ratio)
    df["noun_ratio"] = df["text"].apply(noun_ratio)


    # 감정 분석이랑 광고 위치 분석 함수
    df["sentiment_score"] = df["text"].apply(sentiment_score)
    df["sponsored_position_ratio"] = df["text"].apply(sponsored_keyword_position_ratio)
    df["special_exclamation"] = z_score_binarize(df["text"].apply(exclamation_special_score), threshold=1.0)
    df["has_shopping_link"] = df["text"].apply(has_shopping_mall_link)
    df["image_density_log"] = df.apply(image_density_log, axis=1)

    return df




def main():
    input_path = "ai/input.csv"
    output_path = "ai/output.csv"

    df = pd.read_csv(input_path)

    if 'text' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'text'})

    df = analyze_reviews(df)

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"완료: {output_path}")

if __name__ == "__main__":
    main()
