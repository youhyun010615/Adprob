import pandas as pd
import re
from collections import Counter
from konlpy.tag import Okt

# --- 형태소 분석기 초기화 ---
okt = Okt()

# --- 키워드 설정 ---
sponsored_keywords = [
    # 기존 키워드
    "협찬", "제공", "광고", "체험단", "무료 제공", "서포터즈", "후원", "스폰서", "브랜드 지원",
    "홍보", "제품 제공", "지원받음", "지원받은", "무상 제공", "무상으로 제공", "제공받은", "PPL", "지원", "제품 서포트",
    "브랜드 체험", "브랜드 측에서", "원고료", "광고성", "업체 제공", "광고 포함", "광고 문구", "광고 후기",

    # ✅ 확실한 광고/협찬 표현 추가
    "제품을 제공받아", "브랜드로부터", "광고주", "광고성 글", "지원받아 작성", "협찬받아 작성", "제공받아 작성",
    "이 포스팅은", "이 글은", "해당 브랜드로부터", "제품을 무상으로", "업체로부터", "서포트를 받음", "광고 콘텐츠",
    "유료 광고 포함", "유료 협찬 포함", "본 제품은", "광고 목적", "PR 글", "광고 협찬", "제품 협찬", "브랜드 협찬",
    "서포터즈 활동", "제품을 받아보고", "스폰서십", "브랜드에서 보내준", "브랜드측에서 제공", "제품 리뷰 요청을 받아",
    "제작 지원", "홍보용으로 제공", "마케팅 목적", "광고용", "광고 리뷰", "광고 게시", "광고 안내", "광고 문구 포함"
]


positive_keywords = [
    # 기존 키워드 포함
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

    # ✅ 추가 키워드
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
    # --- 일반 단어 ---
    "가격", "가격대", "가격표", "정가", "할인가", "특가", "세일", "할인", "이벤트", "프로모션",
    "저렴", "저가", "고가", "비쌈", "비싸", "가성비", "합리적", "가성비 좋음", "저렴한", "저렴하게",
    "가성비 있는", "가격 대비", "가치 있는 가격", "가격 만족", "가격 만족도", "착한 가격", "혜자", "혜자템",
    "적당한 가격", "가격 괜찮음", "가격 착해요", "가격이 착하다", "비용 부담 없음", "부담 없는 가격", "가격 경쟁력",

    # --- 숫자 + 단위 표현 ---
    "만원", "만 원", "원", "원대", "천원", "천 원", "1만 원", "1만원", "10,000원", "10000원",
    "5천원", "오천원", "오천 원", "1만원대", "2만원대", "3만원대", "4만원대", "5만원대", "10만원대",
    "몇 천원", "몇 만원", "1~2만원", "2~3만원", "3~4만원", "1만원 이하", "1만원 미만", "5천원 이하", "무료",

    # --- 은어/신조어/감탄형 ---
    "가성비 미쳤다", "혜자스러운", "정신 못 차리는 가격", "말도 안 되는 가격", "찐으로 싸다", "찐 혜자", "가격 미침",
    "너무 싸다", "그 가격에 이 정도?", "돈값함", "이게 이 가격?", "이 가격 실화?", "이 가격이면 무조건 사야", "가격 미쳤어요",
    "이 정도 퀄리티에 이 가격?", "가격 듣고 놀람", "가격 보고 깜짝", "거의 공짜", "거저 줌", "미친 가성비", "혜자 가격", "가격 실화냐"
]


shopping_mall_domains = ["smartstore.naver.com", "musinsa.com", "coupang.com", "gmarket.co.kr", "11st.co.kr", "link.coupang"]


negation_patterns = [
    # 광고/협찬 등의 부정 표현
    r"(광고|협찬|스폰서|지원|체험단)\s*(?:을|를)?\s*(?:전혀|일절|일체)?\s*(받지\s*않|안\s*받|않았|받은\s*적\s*없|없어요|없습니다|아닙니다|아니에요|안\s*했|없었|아님|아녀)",

    # 자비 구매 표현 (내돈내산/직접 구매 계열)
    r"(내\s*돈\s*내\s*산|내돈내산|본인\s*구매|자비로\s*구매|직접\s*구매|내\s*돈\s*주고\s*샀|내\s*돈\s*으로\s*직접\s*구매|사비로\s*구매|직접\s*결제|자체\s*구매|내\s*돈\s*지불)",

    # 영어 부정 표현
    r"(no|not|without|zero)\s*(sponsor|sponsorship|ad|advertisement|promotion|support)",

    # 단어형 부정 표현
    r"(무\s*광고|비\s*광고|비\s*협찬|무\s*협찬|언\s*스폰서|노\s*스폰서|비\s*스폰서|노\s*광고|노\s*협찬)",

    # 리뷰 전체를 부정하는 관용형
    r"(광고\s*아니에요|협찬\s*아닙니다|이건\s*광고\s*아님|이건\s*협찬\s*아님|스폰서\s*아닙니다|협찬\s*없이\s*구매)",

    # "내돈내산" 강조 문장 구조
    r"(정말\s*내돈내산|100%\s*내돈내산|내\s*돈\s*주고\s*구매|광고\s*절대\s*아님|진짜\s*내돈내산)"
]






def is_negated_keyword(text, keyword):
    for pattern in negation_patterns:
        if re.search(pattern, text) and keyword in pattern:
            return True
    return False


def morph_sequence_in_text(morph_sequence, morphs):
    for i in range(len(morphs) - len(morph_sequence) + 1):
        if morphs[i:i+len(morph_sequence)] == morph_sequence:
            return True
    return False


def count_keywords_morph(text, keywords):
    text = str(text)
    morphs = okt.morphs(text)
    count = 0
    for kw in keywords:
        if is_negated_keyword(text, kw):
            continue  # 부정 표현인 경우 스킵
        kw_morphs = okt.morphs(kw)  # 키워드 형태소로 변환
        if morph_sequence_in_text(kw_morphs, morphs):
            count += 1
    return count


def count_external_links(text):
    text = str(text)
    url_pattern = r'((https?:\/\/)?(www\.)?[\w\-]+\.(com|co\.kr|kr|net|org|shop|me|info|biz|tv|store)(\/\S*)?)'
    return len(re.findall(url_pattern, text))



def has_shopping_mall_link(text):
    return int(any(domain in text for domain in shopping_mall_domains))


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


def adj_adv_ratio(text):
    pos_tags = okt.pos(str(text))
    if not pos_tags:
        return 0
    count = sum(1 for word, tag in pos_tags if tag in ["Adjective", "Adverb"])
    return count / len(pos_tags)


def emoji_count(text):
    return len(re.findall(r"[😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰😗😙😚☺🙂🤗🤩🤔🤨😐😑😶🙄😏😣😥😮🤐😯😪😫😴😌😛😜😝🤤😒😓😔😕🙃🤑😲☹🙁😖😞😟😤😢😭😦😧😨😩🤯😬😰😱🥵🥶😳🤪😵😡😠🤬😷🤒🤕🤢🤮🤧😇🤠🥳🥴🥺🤥🤫🤭🧐🤓😈👿🤡👹👺💀☠👻👽👾🤖💩😺😸😹😻😼😽🙀😿😾🙈🙉🙊👶🧒👦👧🧑👱👨👱‍♂🧔👩👱‍♀🧓👴👵👨‍⚕👩‍⚕👨‍🎓👩‍🎓👨‍🏫👩‍🏫👨‍⚖👩‍⚖👨‍🌾👩‍🌾👨‍🍳👩‍🍳👨‍🔧👩‍🔧👨‍🏭👩‍🏭👨‍💼👩‍💼👨‍🔬👩‍🔬👨‍💻👩‍💻👨‍🎤👩‍🎤👨‍🎨👩‍🎨👨‍✈👩‍✈👨‍🚀👩‍🚀👨‍🚒👩‍🚒👮👮‍♂👮‍♀🕵🕵‍♂🕵‍♀💂💂‍♂💂‍♀👷👷‍♂👷‍♀🤴👸👳👳‍♂👳‍♀👲🧕🤵👰🤰🤱👼🎅🤶🦸🦸‍♀🦸‍♂🦹🦹‍♀🦹‍♂🧙🧙‍♀🧙‍♂🧚🧚‍♀🧚‍♂🧛🧛‍♀🧛‍♂🧜🧜‍♀🧜‍♂🧝🧝‍♀🧝‍♂🧞🧞‍♀🧞‍♂🧟🧟‍♀🧟‍♂🙍🙍‍♂🙍‍♀🙎🙎‍♂🙎‍♀🙅🙅‍♂🙅‍♀🙆🙆‍♂🙆‍♀💁💁‍♂💁‍♀🙋🙋‍♂🙋‍♀🙇🙇‍♂🙇‍♀🤦🤦‍♂🤦‍♀🤷🤷‍♂🤷‍♀💆💆‍♂💆‍♀💇💇‍♂💇‍♀🚶🚶‍♂🚶‍♀🏃🏃‍♂🏃‍♀💃🕺👯👯‍♂👯‍♀🧖🧖‍♀🧖‍♂🧗🧗‍♀🧗‍♂🧘🧘‍♀🧘‍♂🛀🛌🕴🗣👤👥🤺🏇⛷🏂🏌🏌‍♂🏌‍♀🏄🏄‍♂🏄‍♀🚣🚣‍♂🚣‍♀🏊🏊‍♂🏊‍♀⛹⛹‍♂⛹‍♀🏋🏋‍♂🏋‍♀🚴🚴‍♂🚴‍♀🚵🚵‍♂🚵‍♀🏎🏍🤸🤸‍♂🤸‍♀🤼🤼‍♂🤼‍♀🤽🤽‍♂🤽‍♀🤾🤾‍♂🤾‍♀🤹🤹‍♂🤹‍♀👫👬👭💏💑👪🤳💪🦵🦶👈👉☝👆🖕👇✌🤞🖖🤘🤙🖐✋👌👍👎✊👊🤛🤜🤚👋🤟✍👏👐🙌🤲🙏🤝💅👂👃👣👀👁👁‍🗨🧠🦴🦷👅👄🦰🦱🦳🦲💋💘💝💖💗💓💞💕💌❣💔❤🧡💛💚💙💜🖤💟💤💢💣💥💦💨💫💬🗨🗯💭🕳👓🕶🥽🥼👔👕👖🧣🧤🧥🧦👗👘👙👚👛👜👝🛍🎒👞👟🥾🥿👠👡👢👑👒🎩🎓🧢⛑📿💄💍💎🐵🐒🦍🐶🐕🐩🐺🦊🦝🐱🐈🦁🐯🐅🐆🐴🐎🦄🦓🦌🐮🐂🐃🐄🐷🐖🐗🐽🐏🐑🐐🐪🐫🦙🦒🐘🦏🦛🐭🐁🐀🐹🐰🐇🐿🦔🦇🐻🐨🐼🦘🦡🐾🦃🐔🐓🐣🐤🐥🐦🐧🕊🦅🦆🦢🦉🦚🦜🐸🐊🐢🦎🐍🐲🐉🦕🦖🐳🐋🐬🐟🐠🐡🦈🐙🐚🦀🦞🦐🦑🐌🦋🐛🐜🐝🐞🦗🕷🕸🦂🦟🦠💐🌸💮🏵🌹🥀🌺🌻🌼🌷🌱🌲🌳🌴🌵🌾🌿☘🍀🍁🍂🍃🍇🍈🍉🍊🍋🍌🍍🥭🍎🍏🍐🍑🍒🍓🥝🍅🥥🥑🍆🥔🥕🌽🌶🥒🥬🥦🍄🥜🌰🍞🥐🥖🥨🥯🥞🧀🍖🍗🥩🥓🍔🍟🍕🌭🥪🌮🌯🥙🥚🍳🥘🍲🥣🥗🍿🧂🥫🍱🍘🍙🍚🍛🍜🍝🍠🍢🍣🍤🍥🥮🍡🥟🥠🥡🍦🍧🍨🍩🍪🎂🍰🧁🥧🍫🍬🍭🍮🍯🍼🥛☕🍵🍶🍾🍷🍸🍹🍺🍻🥂🥃🥤🥢🍽🍴🥄🔪🏺🌍🌎🌏🌐🗺🗾🧭🏔⛰🌋🗻🏕🏖🏜🏝🏞🏟🏛🏗🧱🏘🏚🏠🏡🏢🏣🏤🏥🏦🏨🏩🏪🏫🏬🏭🏯🏰💒🗼🗽⛪🕌🕍⛩🕋⛲⛺🌁🌃🏙🌄🌅🌆🌇🌉♨🌌🎠🎡🎢💈🎪🚂🚃🚄🚅🚆🚇🚈🚉🚊🚝🚞🚋🚌🚍🚎🚐🚑🚒🚓🚔🚕🚖🚗🚘🚙🚚🚛🚜🚲🛴🛹🛵🚏🛣🛤🛢⛽🚨🚥🚦🛑🚧⚓⛵🛶🚤🛳⛴🛥🚢✈🛩🛫🛬💺🚁🚟🚠🚡🛰🚀🛸🛎🧳⌛⏳⌚⏰⏱⏲🕰🕛🕧🕐🕜🕑🕝🕒🕞🕓🕟🕔🕠🕕🕡🕖🕢🕗🕣🕘🕤🕙🕥🕚🕦🌑🌒🌓🌔🌕🌖🌗🌘🌙🌚🌛🌜🌡☀🌝🌞⭐🌟🌠☁⛅⛈🌤🌥🌦🌧🌨🌩🌪🌫🌬🌀🌈🌂☂☔⛱⚡❄☃⛄☄🔥💧🌊🎃🎄🎆🎇🧨✨🎈🎉🎊🎋🎍🎎🎏🎐🎑🧧🎀🎁🎗🎟🎫🎖🏆🏅🥇🥈🥉⚽⚾🥎🏀🏐🏈🏉🎾🥏🎳🏏🏑🏒🥍🏓🏸🥊🥋🥅⛳⛸🎣🎽🎿🛷🥌🎯🎱🔮🧿🎮🕹🎰🎲🧩🧸♠♥♦♣♟🃏🀄🎴🎭🖼🎨🧵🧶🔇🔈🔉🔊📢📣📯🔔🔕🎼🎵🎶🎙🎚🎛🎤🎧📻🎷🎸🎹🎺🎻🥁📱📲☎📞📟📠🔋🔌💻🖥🖨⌨🖱🖲💽💾💿📀🧮🎥🎞📽🎬📺📷📸📹📼🔍🔎🕯💡🔦🏮📔📕📖📗📘📙📚📓📒📃📜📄📰🗞📑🔖🏷💰💴💵💶💷💸💳🧾💹💱💲✉📧📨📩📤📥📦📫📪📬📭📮🗳✏✒🖋🖊🖌🖍📝💼📁📂🗂📅📆🗒🗓📇📈📉📊📋📌📍📎🖇📏📐✂🗃🗄🗑🔒🔓🔏🔐🔑🗝🔨⛏⚒🛠🗡⚔🔫🏹🛡🔧🔩⚙🗜⚖🔗⛓🧰🧲⚗🧪🧫🧬🔬🔭📡💉💊🚪🛏🛋🚽🚿🛁🧴🧷🧹🧺🧻🧼🧽🧯🛒🚬⚰⚱🗿🏧🚮🚰♿🚹🚺🚻🚼🚾🛂🛃🛄🛅⚠🚸⛔🚫🚳🚭🚯🚱🚷📵🔞☢☣⬆↗➡↘⬇↙⬅↖↕↔↩↪⤴⤵🔃🔄🔙🔚🔛🔜🔝🛐⚛🕉✡☸☯✝☦☪☮🕎🔯♈♉♊♋♌♍♎♏♐♑♒♓⛎🔀🔁🔂▶⏩⏭⏯◀⏪⏮🔼⏫🔽⏬⏸⏹⏺⏏🎦🔅🔆📶📳📴♀♂⚕♾♻⚜🔱📛🔰⭕✅☑✔✖❌❎➕➖➗➰➿〽✳✴❇‼⁉❓❔❕❗〰©®™💯🔠🔡🔢🔣🔤🔴🔵⚪⚫⬜⬛]", str(text)))


def first_person_ratio(text):
    first_person_pronouns = ["나", "내", "저", "저는", "제가", "저도", "내가"]
    words = okt.morphs(str(text))
    if not words:
        return 0
    count = sum(w in first_person_pronouns for w in words)
    return count / len(words)


def exclamation_question_ratio(text):
    text = str(text)
    if not text:
        return 0
    count = text.count("!") + text.count("?")
    return count / len(text)

def special_char_ratio(text):
    text = str(text)
    if not text:
        return 0
    count = len(re.findall(r"[~^]", text))
    return count / len(text)





# --- 분석 함수 -------------------------------------------------------------
def analyze_reviews(df):
    df["sponsored_keywords_count"] = df["text"].apply(lambda x: count_keywords_morph(x, sponsored_keywords))
    df["positive_bias_keywords"] = df["text"].apply(lambda x: count_keywords_morph(x, positive_keywords))
    df["external_links_count"] = df["text"].apply(count_external_links)
    df["shopping_mall_linked"] = df["text"].apply(has_shopping_mall_link)
    df["text_length"] = df["text"].apply(lambda x: len(str(x)))
    df["most_common_word_freq_ratio"] = df["text"].apply(count_most_common_word_freq_ratio)
    df["price_mentions"] = df["text"].apply(lambda x: count_keywords_morph(x, price_keywords))
    df["adj_adv_ratio"] = df["text"].apply(adj_adv_ratio)
    df["has_emoji"] = df["text"].apply(lambda x: int(emoji_count(x) > 0))
    df["first_person_ratio"] = df["text"].apply(first_person_ratio)
    df["exclamation_question_ratio"] = df["text"].apply(exclamation_question_ratio)
    df["special_char_ratio"] = df["text"].apply(special_char_ratio)




    return df

# --- 실행 ---
def main():
    input_path = (R"C:\Users\lgrr-\Downloads\25졸프\데이터 모으기\feature\input.csv")
    output_path = "output.csv"

    df = pd.read_excel(input_path)
    df = pd.DataFrame(df.iloc[:, 0])
    df.columns = ["text"]


    df["image_count"] = 0

    df = analyze_reviews(df)
   
    df["is_ad"] = 0



    df.to_excel(output_path, index=False)
    print(f"분석 완료: {output_path}")

main()