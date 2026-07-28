import base64
import json
import os
import random
import re
from typing import Any

import requests
import streamlit as st

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
    "body_type",
    "top",
    "bottom",
    "shoes",
    "hat",
    "hat_type",
    "hat_color",
    "accessories",
    "hair",
    "special_features",
    "gender_en",
    "age_en",
    "body_type_en",
    "top_en",
    "bottom_en",
    "shoes_en",
    "hat_en",
    "hat_type_en",
    "hat_color_en",
    "accessories_en",
]

HAT_TYPE_MAP = {
    "캡모자": "baseball cap",
    "야구모자": "baseball cap",
    "캡": "baseball cap",
    "비니": "beanie",
    "벙거지": "bucket hat",
    "버킷햇": "bucket hat",
    "챙모자": "brimmed hat",
    "모자": "hat",
}


def get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, "")).strip()


def cf_url(model: str) -> str:
    account_id = get_secret("CLOUDFLARE_ACCOUNT_ID")
    if not account_id:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID가 없습니다.")
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"


def cf_headers() -> dict:
    token = get_secret("CLOUDFLARE_API_TOKEN")
    if not token:
        raise RuntimeError("CLOUDFLARE_API_TOKEN이 없습니다.")
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
        msg = errors[0].get("message") if errors else str(data)
        raise RuntimeError(
            f"Cloudflare AI 요청 실패 (HTTP {response.status_code}): {msg}"
        )

    return data.get("result")


def extract_features(message: str) -> dict:
    schema = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in FIELDS},
        "required": FIELDS,
        "additionalProperties": False,
    }

    system_prompt = """
너는 대한민국 실종 재난문자에서 인상착의를 추출하는 시스템이다.

절대 규칙:
1. 원문에 실제로 적힌 정보만 추출한다.
2. 없는 정보는 반드시 빈 문자열("")로 둔다.
3. 모자가 있으면 hat에 기록하고, 종류가 있으면 hat_type에 기록한다.
4. 예시: 캡모자/야구모자/캡 -> hat_type은 "캡모자"
5. 예시: 비니 -> hat_type은 "비니"
6. 예시: 벙거지/버킷햇 -> hat_type은 "벙거지"
7. hat_color에는 모자 색상을 적는다.
8. 상의, 하의, 신발의 색상과 종류를 절대로 바꾸지 않는다.
9. 영어 필드는 한국어 필드를 정확히 번역한다.
10. 검정색 긴바지 -> black long pants
11. 회색 캡모자 -> gray baseball cap
12. 검정색 크록스 -> black Crocs
"""

    result = cloudflare_request(
        TEXT_MODEL,
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"다음 실종 재난문자를 구조화해줘:\n\n{message}",
                },
            ],
            "temperature": 0.0,
            "max_tokens": 1200,
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
        raise RuntimeError("인상착의 분석 결과가 올바른 형식이 아닙니다.")

    cleaned = {key: str(parsed.get(key, "") or "").strip() for key in FIELDS}

    if cleaned["hat"] and not cleaned["hat_type"]:
        if "캡" in cleaned["hat"] or "야구모자" in cleaned["hat"]:
            cleaned["hat_type"] = "캡모자"
        elif "비니" in cleaned["hat"]:
            cleaned["hat_type"] = "비니"
        elif "벙거지" in cleaned["hat"] or "버킷햇" in cleaned["hat"]:
            cleaned["hat_type"] = "벙거지"
        elif "모자" in cleaned["hat"]:
            cleaned["hat_type"] = "모자"

    if cleaned["hat_type"] and not cleaned["hat_type_en"]:
        cleaned["hat_type_en"] = HAT_TYPE_MAP.get(cleaned["hat_type"], "hat")

    return cleaned


def required_items(features: dict) -> list[str]:
    items = []

    mapping = [
        ("gender_en", "person"),
        ("age_en", "age"),
        ("body_type_en", "body build"),
        ("top_en", "TOP"),
        ("bottom_en", "BOTTOM"),
        ("shoes_en", "SHOES"),
        ("hat_en", "HAT"),
        ("hat_type_en", "HAT_TYPE"),
        ("accessories_en", "ACCESSORIES"),
    ]

    for key, label in mapping:
        value = features.get(key, "").strip()
        if value:
            items.append(f"{label}: {value}")

    return items


def build_generation_prompt(features: dict, correction: str = "") -> str:
    must = required_items(features)

    if not must:
        raise RuntimeError("이미지로 만들 수 있는 인상착의 정보가 없습니다.")

    mandatory = "\n".join(f"- {item}" for item in must)

    hat_rule = ""
    if features.get("hat_type"):
        hat_type_kr = features["hat_type"]
        hat_type_en = features.get("hat_type_en") or "hat"
        hat_rule = (
            f"- REQUIRED HAT TYPE: {hat_type_en}. "
            f"A generic different hat is wrong. "
            f"If the required type is {hat_type_en}, do not use another hat type.\n"
        )

    correction_text = ""
    if correction:
        correction_text = f"""
이전 이미지에서 잘못된 점:
{correction}
위 문제만 정확히 수정하고, 이미 맞았던 요소는 유지해라.
"""

    return f"""
Create exactly ONE full-body missing-person appearance reference illustration.

MANDATORY REQUIREMENTS — EVERY ITEM BELOW MUST BE VISIBLE AND CORRECT:
{mandatory}

CRITICAL RULES:
- Show the complete body from head to feet.
- Front-facing natural standing pose.
- Clothing COLORS must match exactly.
- Clothing TYPES must match exactly.
- If a hat is listed, the hat MUST be clearly visible on the person's head.
{hat_rule}- Shoes must be clearly visible and match exactly.
- Accessories must be visible when listed.
- Do not substitute similar colors.
- BLACK means BLACK, not white, gray, blue, beige, or navy.
- Do not replace long pants with shorts.
- Do not replace Crocs with sneakers.
- Plain light studio background.
- Image must contain NO TEXT at all.
- No Korean letters, no English letters, no Chinese letters, no Japanese letters.
- No labels, no captions, no posters, no signs, no watermarks, no symbols, no numbers.
- Only a single person should appear.
- Do not invent distinctive facial features that were not provided.

{correction_text}
""".strip()


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

    b64 = result["image"]
    return base64.b64decode(b64), b64


def extract_text_from_result(result: Any) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        for key in ("response", "result", "text", "answer", "caption"):
            value = result.get(key)
            if isinstance(value, str):
                return value

    return json.dumps(result, ensure_ascii=False)


def moondream_query(image_b64: str, prompt: str) -> str:
    data_uri = f"data:image/jpeg;base64,{image_b64}"

    payloads = [
        {
            "task": "query",
            "image": data_uri,
            "prompt": prompt,
            "max_tokens": 700,
        },
        {
            "image": data_uri,
            "prompt": prompt,
            "max_tokens": 700,
        },
    ]

    last_error = None
    for payload in payloads:
        try:
            result = cloudflare_request(VISION_MODEL, payload)
            return extract_text_from_result(result)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"이미지 검수 AI 호출 실패: {last_error}")


def parse_json_loose(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {}


def verify_image(image_b64: str, features: dict) -> dict:
    requirements = "\n".join(f"- {item}" for item in required_items(features))
    hat_note = ""
    if features.get("hat_type_en"):
        hat_note = (
            f"\nThe required hat type is exactly: {features['hat_type_en']}."
            f"\nA generic hat is NOT enough."
        )

    prompt = f"""
Inspect this generated full-body person image very carefully.

EXPECTED APPEARANCE:
{requirements}
{hat_note}

Check especially:
1. top clothing color and type
2. bottom clothing color and type
3. shoes color and type
4. hat presence, hat type, and hat color
5. listed accessories
6. whether the full body is visible
7. whether ANY text exists inside the image

Be strict.
If black pants are required, white/gray/navy pants are wrong.
If a gray baseball cap is required, a beanie or a generic hat is wrong.
If there is any text in the image, that is wrong.

Return JSON ONLY in this exact structure:
{{
  "score": 0,
  "pass": false,
  "missing": [],
  "wrong": [],
  "has_text": false,
  "feedback_en": ""
}}

score must be 0-100.
pass can be true only if all mandatory items match and has_text is false.
feedback_en must be a short English correction instruction for the image generator.
""".strip()

    raw = moondream_query(image_b64, prompt)
    parsed = parse_json_loose(raw)

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
    feedback = str(parsed.get("feedback_en", "") or "").strip()
    passed = bool(parsed.get("pass", False))

    if has_text:
        wrong.append("이미지 안에 글자가 들어감")

    if not parsed:
        passed = False
        feedback = (
            "Regenerate the image and obey every mandatory clothing, hat, shoe, "
            "color, and accessory requirement exactly. Ensure absolutely no text is present."
        )

    return {
        "score": max(0, min(score, 100)),
        "pass": passed and not missing and not wrong and not has_text,
        "missing": missing,
        "wrong": wrong,
        "has_text": has_text,
        "feedback_en": feedback,
        "raw": raw,
    }


def safe(value: str) -> str:
    return value if value else "정보 없음"


st.title("🔎 FindVision AI")
st.caption(
    "실종 재난문자를 분석하고, 생성 이미지가 인상착의와 맞는지 AI가 다시 검사합니다."
)

st.warning(
    "이 이미지는 실제 얼굴 복원이 아니라 인상착의 참고용입니다. "
    "생성형 AI 특성상 100% 일치를 보장할 수 없으며, 앱은 최대 3회 자동 검수·재생성을 수행합니다."
)

sample = (
    "실종자 남성 68세, 키 164cm, 58kg, "
    "빨간색 반팔티, 검정색 긴바지, 검정색 크록스, 회색 캡모자 착용"
)

message = st.text_area(
    "실종 재난문자 입력",
    value=sample,
    height=150,
)

if st.button(
    "AI 분석 → 이미지 생성 → 자동 검수",
    type="primary",
    use_container_width=True,
):
    if not message.strip():
        st.warning("실종 재난문자를 입력해 주세요.")
        st.stop()

    try:
        with st.spinner("1/3 인상착의 정보를 정확하게 추출하고 있습니다..."):
            features = extract_features(message.strip())

        st.subheader("추출된 인상착의")
        c1, c2 = st.columns(2)

        labels = [
            ("name", "이름"),
            ("gender", "성별"),
            ("age", "나이"),
            ("height", "키"),
            ("weight", "몸무게"),
            ("body_type", "체형"),
            ("top", "상의"),
            ("bottom", "하의"),
            ("shoes", "신발"),
            ("hat", "모자"),
            ("hat_type", "모자 종류"),
            ("hat_color", "모자 색상"),
            ("accessories", "소지품/액세서리"),
            ("hair", "머리"),
            ("special_features", "기타 특징"),
        ]

        for i, (key, label) in enumerate(labels):
            target = c1 if i < 8 else c2
            target.write(f"**{label}:** {safe(features.get(key, ''))}")

        best = None
        correction = ""

        progress = st.progress(0)
        status = st.empty()

        for attempt in range(1, MAX_ATTEMPTS + 1):
            status.info(f"{attempt}차 이미지 생성 중...")
            prompt = build_generation_prompt(features, correction)

            image_bytes, image_b64 = generate_image(prompt)

            progress.progress(int((attempt - 0.5) / MAX_ATTEMPTS * 100))
            status.info(f"{attempt}차 이미지가 인상착의와 맞는지 검사 중...")

            verification = verify_image(image_b64, features)

            candidate = {
                "attempt": attempt,
                "image": image_bytes,
                "prompt": prompt,
                "verification": verification,
            }

            if best is None or verification["score"] > best["verification"]["score"]:
                best = candidate

            if verification["pass"]:
                best = candidate
                break

            correction = verification["feedback_en"] or (
                "Regenerate and strictly correct all missing or wrong mandatory items. "
                "Also ensure there is absolutely no text inside the image."
            )

            progress.progress(int(attempt / MAX_ATTEMPTS * 100))

        progress.progress(100)
        status.empty()

        if best is None:
            raise RuntimeError("이미지를 생성하지 못했습니다.")

        verdict = best["verification"]

        st.divider()
        st.subheader("최종 참고 이미지")
        st.image(
            best["image"],
            caption=f"{best['attempt']}차 생성 결과 · 검수 점수 {verdict['score']}/100",
            use_container_width=True,
        )

        if verdict["pass"]:
            st.success(
                f"자동 검수 통과: {best['attempt']}차 생성 이미지가 필수 인상착의 조건을 충족했습니다."
            )
        else:
            st.warning(
                "3회 안에 모든 조건을 완전히 통과하지 못해서 가장 높은 점수의 이미지를 표시했습니다. "
                "실제 사용 전 사람이 한 번 더 확인해야 합니다."
            )

        if verdict["missing"]:
            st.write("**누락된 항목:**", ", ".join(map(str, verdict["missing"])))

        if verdict["wrong"]:
            st.write("**잘못 표현된 항목:**", ", ".join(map(str, verdict["wrong"])))

        if verdict["has_text"]:
            st.error("이미지 안에 글자가 들어간 것으로 판단되어 재생성 대상이 되었습니다.")

        with st.expander("원문 재난문자 보기"):
            st.write(message)

    except Exception as exc:
        st.error(f"오류가 발생했습니다: {exc}")
