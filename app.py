import base64
import json
import os
import random
import re
from typing import Any

import requests
import streamlit as st


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="FindVision AI",
    page_icon="🔎",
    layout="wide",
)

TEXT_MODEL = "@cf/meta/llama-3.1-8b-instruct-fast"
IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
VISION_MODEL = "@cf/moondream/moondream3.1-9B-A2B"

MAX_ATTEMPTS = 3

FIELDS = [
    "name",
    "gender",
    "age",
    "height",
    "weight",
    "nationality",
    "skin_tone",
    "hair_color",
    "hair_length",
    "hair_texture",
    "top",
    "bottom",
    "shoes",
    "hat_type",
    "hat_color",
    "glasses",
    "facial_hair",
    "accessories",
    "special_features",
    "image_prompt_en",
    "ambiguity_notes",
]

# 이미지 생성에 실제로 유용한 인상착의 항목
APPEARANCE_FIELDS = [
    "skin_tone",
    "hair_color",
    "hair_length",
    "hair_texture",
    "top",
    "bottom",
    "shoes",
    "hat_type",
    "hat_color",
    "glasses",
    "facial_hair",
    "accessories",
    "special_features",
]

LABELS = {
    "name": "이름",
    "gender": "성별",
    "age": "나이",
    "height": "키",
    "weight": "몸무게",
    "nationality": "국적",
    "skin_tone": "피부톤",
    "hair_color": "머리색",
    "hair_length": "머리 길이",
    "hair_texture": "머리 형태",
    "top": "상의",
    "bottom": "하의",
    "shoes": "신발",
    "hat_type": "모자 종류",
    "hat_color": "모자 색상",
    "glasses": "안경",
    "facial_hair": "수염",
    "accessories": "소지품·액세서리",
    "special_features": "기타 특징",
}


# =========================================================
# Cloudflare API
# =========================================================

def get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, "")).strip()


def cf_url(model: str) -> str:
    account_id = get_secret("CLOUDFLARE_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("Cloudflare Account ID가 설정되어 있지 않습니다.")

    return (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{model}"
    )


def cf_headers() -> dict:
    token = get_secret("CLOUDFLARE_API_TOKEN")
    if not token:
        raise RuntimeError("Cloudflare API Token이 설정되어 있지 않습니다.")

    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def cloudflare_request(model: str, payload: dict, timeout: int = 120) -> Any:
    response = requests.post(
        cf_url(model),
        headers=cf_headers(),
        json=payload,
        timeout=timeout,
    )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Cloudflare 응답을 읽지 못했습니다. HTTP {response.status_code}"
        ) from exc

    if not response.ok or not data.get("success", False):
        errors = data.get("errors") or []
        message = errors[0].get("message") if errors else str(data)

        raise RuntimeError(
            f"Cloudflare AI 요청 실패 [{model}] "
            f"(HTTP {response.status_code}): {message}"
        )

    return data.get("result")


# =========================================================
# 재난문자 분석
# =========================================================

def extract_features(message: str) -> dict:
    schema = {
        "type": "object",
        "properties": {
            key: {"type": "string"}
            for key in FIELDS
        },
        "required": FIELDS,
        "additionalProperties": False,
    }

    system_prompt = """
너는 대한민국 실종 재난문자의 '인상착의 사실 추출기'다.

가장 중요한 원칙은 절대로 없는 정보를 만들어내지 않는 것이다.

[추출 규칙]
1. 원문에 명확하게 존재하는 정보만 추출한다.
2. 명확하지 않거나 없는 정보는 반드시 빈 문자열("")로 둔다.
3. 이름만 보고 국적, 피부톤, 머리 모양 등을 추측하지 않는다.
4. '모자'라고만 적혀 있으면 hat_type은 '모자(종류 불명)'으로 기록한다.
5. '캡모자', '야구모자', '비니', '벙거지', '버킷햇'처럼 종류가 명시된 경우에만 정확한 종류를 기록한다.
6. '검은 바지'라고만 적혀 있으면 긴바지/반바지를 임의로 정하지 않는다.
7. 피부톤이 원문에 없으면 피부톤을 추측하지 않는다.
8. 곱슬/직모/파마 등 머리 형태가 없으면 임의로 정하지 않는다.
9. 국적이 명시되어 있으면 nationality에 기록하고, 없으면 빈 문자열로 둔다.
10. 안경, 수염, 액세서리도 원문에 있는 경우에만 기록한다.
11. ambiguity_notes에는 원문에서 애매해서 구체적으로 정할 수 없는 부분을 한국어로 짧게 적는다.
12. image_prompt_en은 이미지 생성용 영어 설명이다.
13. image_prompt_en에도 원문에 실제로 있는 특징만 넣는다.
14. image_prompt_en에 실제 얼굴 생김새를 임의로 추가하지 않는다.
15. 색상과 복장 종류는 원문의 내용을 정확히 유지한다.

예:
"검은색 모자" ->
hat_type: "모자(종류 불명)"
hat_color: "검은색"

"회색 캡모자" ->
hat_type: "캡모자"
hat_color: "회색"

"검은색 바지" ->
bottom: "검은색 바지"
(긴바지라고 추측하면 안 됨)
""".strip()

    result = cloudflare_request(
        TEXT_MODEL,
        {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "다음 실종 재난문자를 분석해 주세요.\n\n"
                        f"{message}"
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 1200,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": schema,
            },
        },
    )

    if isinstance(result, dict):
        parsed = result.get("response", result)
    else:
        parsed = result

    if isinstance(parsed, str):
        parsed = json.loads(parsed)

    if not isinstance(parsed, dict):
        raise RuntimeError("AI 분석 결과가 올바른 형식이 아닙니다.")

    return {
        key: str(parsed.get(key, "") or "").strip()
        for key in FIELDS
    }


# =========================================================
# 기본 인물 설정
# =========================================================

def get_origin(features: dict) -> tuple[str, str]:
    nationality = features.get("nationality", "").strip()

    # 국적이 명시됐으면 원문 정보 우선
    if nationality:
        return nationality, nationality

    # 프로젝트 기본 설정
    return "한국인", "Korean"


# =========================================================
# 상세도 검사
# =========================================================

def get_known_appearance_count(features: dict) -> int:
    return sum(
        1 for key in APPEARANCE_FIELDS
        if features.get(key, "").strip()
    )


def get_missing_recommended(features: dict) -> list[str]:
    recommended = [
        ("top", "상의"),
        ("bottom", "하의"),
        ("shoes", "신발"),
        ("hair_texture", "머리 형태"),
        ("skin_tone", "피부톤"),
    ]

    return [
        label
        for key, label in recommended
        if not features.get(key, "").strip()
    ]


# =========================================================
# 이미지 생성 프롬프트
# =========================================================

def build_generation_prompt(
    features: dict,
    origin_en: str,
    correction: str = "",
) -> str:

    facts = []

    def add(label: str, key: str):
        value = features.get(key, "").strip()
        if value:
            facts.append(f"- {label}: {value}")

    add("Gender", "gender")
    add("Age", "age")
    add("Height", "height")
    add("Weight", "weight")
    add("Skin tone", "skin_tone")
    add("Hair color", "hair_color")
    add("Hair length", "hair_length")
    add("Hair texture/style", "hair_texture")
    add("Top clothing", "top")
    add("Bottom clothing", "bottom")
    add("Shoes", "shoes")
    add("Hat type", "hat_type")
    add("Hat color", "hat_color")
    add("Glasses", "glasses")
    add("Facial hair", "facial_hair")
    add("Accessories", "accessories")
    add("Other visible features", "special_features")

    facts_text = "\n".join(facts)

    correction_text = ""
    if correction:
        correction_text = f"""
THE PREVIOUS IMAGE FAILED VERIFICATION.
Correct these mistakes:
{correction}

Keep every already-correct requirement unchanged.
"""

    prompt = f"""
Create exactly ONE realistic full-body appearance reference illustration.

PERSON:
- Origin/nationality setting: {origin_en}

KNOWN FACTS:
{facts_text}

STRICT RULES:
- Use ONLY the known facts above.
- Do not invent distinctive facial features.
- Do not invent a hat type that was not specified.
- Do not invent a clothing length or type that was not specified.
- Clothing colors must match the known facts exactly.
- Clothing types must match the known facts exactly.
- If a specific hat type is stated, show that exact type.
- If shoes are stated, show the correct shoe type and color.
- If skin tone or hair characteristics are stated, reflect them accurately.
- Show the complete body from head to feet.
- One person only.
- Front-facing natural standing pose.
- Plain light studio background.
- NO TEXT anywhere in the image.
- NO Korean, English, Chinese, Japanese, numbers, labels, captions, signs, posters or watermarks.

{correction_text}
""".strip()

    # FLUX.1 Schnell prompt 최대 길이를 고려하여 안전하게 제한
    return prompt[:2000]


def generate_image(prompt: str) -> tuple[bytes, str]:
    result = cloudflare_request(
        IMAGE_MODEL,
        {
            "prompt": prompt,
            "steps": 8,
            "seed": random.randint(1, 999_999_999),
        },
    )

    if not isinstance(result, dict) or not result.get("image"):
        raise RuntimeError("이미지 생성 결과를 받지 못했습니다.")

    image_b64 = result["image"]
    return base64.b64decode(image_b64), image_b64


# =========================================================
# Vision AI 검수
# =========================================================

def extract_text_from_result(result: Any) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        for key in (
            "answer",
            "response",
            "result",
            "text",
            "caption",
        ):
            value = result.get(key)
            if isinstance(value, str):
                return value

    return json.dumps(result, ensure_ascii=False)


def moondream_query(image_b64: str, question: str) -> str:
    data_uri = f"data:image/jpeg;base64,{image_b64}"

    result = cloudflare_request(
        VISION_MODEL,
        {
            "task": "query",
            "image": data_uri,
            "question": question,
            "reasoning": False,
            "temperature": 0.0,
            "max_tokens": 800,
            "stream": False,
        },
    )

    return extract_text_from_result(result)


def parse_json_loose(text: str) -> dict:
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.S)

    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {}


def build_verification_requirements(features: dict) -> str:
    lines = []

    for key in APPEARANCE_FIELDS:
        value = features.get(key, "").strip()
        if value:
            lines.append(f"- {LABELS[key]}: {value}")

    return "\n".join(lines)


def verify_image(image_b64: str, features: dict) -> dict:
    requirements = build_verification_requirements(features)

    question = f"""
You are verifying an AI-generated missing-person appearance reference image.

Only verify requirements that are explicitly listed below.
Do NOT penalize unspecified details.

REQUIRED VISIBLE FACTS:
{requirements}

Check:
1. clothing colors
2. clothing types
3. shoe type/color
4. exact hat type/color when specified
5. hair characteristics when specified
6. skin tone when specified
7. glasses/facial hair/accessories when specified
8. full body visibility
9. whether any text, letters, numbers, signs, captions or watermark appear

Be strict about explicitly stated facts.
For example:
- If "캡모자" is required, another hat type is wrong.
- If only generic "모자(종류 불명)" is given, do NOT require a specific hat style.
- If the message only says "바지", do NOT require long pants or shorts.
- Unspecified face details must NOT affect the score.

Return JSON ONLY:

{{
  "score": 0,
  "pass": false,
  "missing": [],
  "wrong": [],
  "has_text": false,
  "feedback_en": ""
}}

score: integer 0-100.
pass: true only when all explicitly specified visible facts are satisfied and has_text is false.
feedback_en: short English instructions to fix only the failed requirements.
""".strip()

    raw = moondream_query(image_b64, question)
    parsed = parse_json_loose(raw)

    if not parsed:
        return {
            "score": 0,
            "pass": False,
            "missing": [],
            "wrong": ["검수 결과를 구조화하지 못함"],
            "has_text": False,
            "feedback_en": (
                "Regenerate while strictly following all explicitly stated "
                "appearance requirements. Do not add any text."
            ),
            "raw": raw,
        }

    try:
        score = int(parsed.get("score", 0))
    except Exception:
        score = 0

    missing = parsed.get("missing", [])
    wrong = parsed.get("wrong", [])

    if not isinstance(missing, list):
        missing = [str(missing)]

    if not isinstance(wrong, list):
        wrong = [str(wrong)]

    has_text = bool(parsed.get("has_text", False))

    if has_text and "이미지 안에 글자가 있음" not in wrong:
        wrong.append("이미지 안에 글자가 있음")

    feedback = str(parsed.get("feedback_en", "") or "").strip()

    passed = (
        bool(parsed.get("pass", False))
        and not missing
        and not wrong
        and not has_text
    )

    return {
        "score": max(0, min(score, 100)),
        "pass": passed,
        "missing": missing,
        "wrong": wrong,
        "has_text": has_text,
        "feedback_en": feedback,
        "raw": raw,
    }


# =========================================================
# UI
# =========================================================

st.title("🔎 FindVision AI")

st.caption(
    "상세한 실종 재난문자를 입력하면 AI가 인상착의를 분석하고 "
    "전신 참고 이미지를 생성한 뒤 Vision AI가 다시 검수합니다."
)

st.warning(
    "생성 이미지는 실제 실종자의 얼굴을 복원한 사진이 아닙니다. "
    "재난문자에 적힌 인상착의를 이해하기 위한 시각적 참고 자료입니다."
)

with st.expander("📌 FindVision에서 권장하는 상세 재난문자 기준", expanded=True):
    st.markdown(
        """
**가능하면 다음 정보가 포함된 재난문자를 권장합니다.**

- 성별 / 나이
- 키 / 몸무게
- 피부톤
- 머리색 / 머리 길이 / 곱슬·직모 등 머리 형태
- 상의 색상과 종류
- 하의 색상과 종류
- 신발 색상과 종류
- 모자 색상과 정확한 종류
- 안경 / 수염 / 소지품 등 기타 특징

정보가 없는 항목은 AI가 임의로 만들어내지 않도록 설계했습니다.
        """
    )

sample = (
    "실종자 남성 68세, 키 164cm, 몸무게 58kg, "
    "피부는 어두운 편, 짧은 검은색 곱슬머리, "
    "회색 캡모자, 빨간색 반팔티, 검정색 긴바지, "
    "검정색 크록스 착용"
)

message = st.text_area(
    "실종 재난문자 입력",
    value=sample,
    height=170,
    placeholder="상세한 실종 재난문자를 입력해 주세요.",
)

generate = st.button(
    "AI 분석 및 참고 이미지 생성",
    type="primary",
    use_container_width=True,
)

if generate:
    if not message.strip():
        st.warning("실종 재난문자를 입력해 주세요.")
        st.stop()

    try:
        with st.spinner("재난문자의 인상착의를 분석하고 있습니다..."):
            features = extract_features(message.strip())

        origin_kr, origin_en = get_origin(features)

        st.subheader("1. AI가 분석한 인상착의")

        st.write(f"**기본 인물 설정:** {origin_kr}")

        col1, col2 = st.columns(2)

        display_keys = [
            "name",
            "gender",
            "age",
            "height",
            "weight",
            "skin_tone",
            "hair_color",
            "hair_length",
            "hair_texture",
            "top",
            "bottom",
            "shoes",
            "hat_type",
            "hat_color",
            "glasses",
            "facial_hair",
            "accessories",
            "special_features",
        ]

        for index, key in enumerate(display_keys):
            target = col1 if index < len(display_keys) / 2 else col2
            value = features.get(key, "").strip() or "정보 없음"
            target.write(f"**{LABELS[key]}:** {value}")

        ambiguity = features.get("ambiguity_notes", "").strip()

        if ambiguity:
            st.info(
                "⚠️ **AI가 임의로 추측하지 않은 모호한 정보:** "
                + ambiguity
            )

        known_count = get_known_appearance_count(features)
        missing_recommended = get_missing_recommended(features)

        # 너무 빈약한 문장은 생성하지 않음
        if known_count < 3:
            st.error(
                "현재 재난문자에는 이미지를 안정적으로 만들기 위한 "
                "인상착의 정보가 너무 적습니다."
            )

            if missing_recommended:
                st.write(
                    "**추가하면 좋은 정보:** "
                    + ", ".join(missing_recommended)
                )

            st.stop()

        if missing_recommended:
            st.warning(
                "일부 정보가 없습니다: "
                + ", ".join(missing_recommended)
                + ". 없는 정보는 AI가 임의로 추측하지 않습니다."
            )

        st.divider()
        st.subheader("2. 전신 참고 이미지 생성 및 자동 검수")

        best = None
        correction = ""

        progress = st.progress(0)
        status = st.empty()

        for attempt in range(1, MAX_ATTEMPTS + 1):
            status.info(f"{attempt}차 참고 이미지 생성 중...")

            prompt = build_generation_prompt(
                features,
                origin_en,
                correction,
            )

            image_bytes, image_b64 = generate_image(prompt)

            progress.progress(
                int(((attempt - 0.5) / MAX_ATTEMPTS) * 100)
            )

            status.info(
                f"{attempt}차 이미지가 입력 정보와 맞는지 검수 중..."
            )

            verification = verify_image(
                image_b64,
                features,
            )

            candidate = {
                "attempt": attempt,
                "image": image_bytes,
                "verification": verification,
            }

            if (
                best is None
                or verification["score"]
                > best["verification"]["score"]
            ):
                best = candidate

            if verification["pass"]:
                best = candidate
                break

            correction = (
                verification["feedback_en"]
                or (
                    "Correct only the failed explicitly stated appearance "
                    "requirements. Do not add any text."
                )
            )

            progress.progress(
                int((attempt / MAX_ATTEMPTS) * 100)
            )

        progress.progress(100)
        status.empty()

        if best is None:
            raise RuntimeError("최종 이미지를 생성하지 못했습니다.")

        verdict = best["verification"]

        st.image(
            best["image"],
            caption=(
                f"{best['attempt']}차 생성 결과 · "
                f"자동 검수 점수 {verdict['score']}/100"
            ),
            use_container_width=True,
        )

        if verdict["pass"]:
            st.success(
                "입력된 인상착의 조건에 대한 자동 검수를 통과했습니다."
            )
        else:
            st.warning(
                "3회 안에 모든 조건을 통과하지 못해 "
                "가장 높은 점수의 이미지를 표시했습니다. "
                "최종 사용 전 사람이 다시 확인해야 합니다."
            )

        if verdict["missing"]:
            st.write(
                "**이미지에서 누락된 항목:** "
                + ", ".join(map(str, verdict["missing"]))
            )

        if verdict["wrong"]:
            st.write(
                "**잘못 표현된 항목:** "
                + ", ".join(map(str, verdict["wrong"]))
            )

        with st.expander("원문 재난문자 보기"):
            st.write(message)

        st.caption(
            "FindVision AI는 원문에 없는 세부 정보를 사실처럼 확정하지 않습니다."
        )

    except Exception as exc:
        st.error(f"오류가 발생했습니다: {exc}")
