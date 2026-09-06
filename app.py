import base64
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
import streamlit as st
from streamlit_cookies_controller import CookieController


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="FindVision AI",
    page_icon="🔎",
    layout="wide",
)

cookie_controller = CookieController()

TEXT_MODEL = "@cf/meta/llama-3.1-8b-instruct-fast"
IMAGE_MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
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
    "verification_requirements_en",
    "ambiguity_notes",
]

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


def auth_header() -> dict:
    token = get_secret("CLOUDFLARE_API_TOKEN")
    if not token:
        raise RuntimeError("Cloudflare API Token이 설정되어 있지 않습니다.")
    return {"Authorization": f"Bearer {token}"}


def json_headers() -> dict:
    headers = auth_header()
    headers["Content-Type"] = "application/json"
    return headers


def cloudflare_json_request(model: str, payload: dict, timeout: int = 120) -> Any:
    response = requests.post(
        cf_url(model),
        headers=json_headers(),
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


def cloudflare_multipart_request(model: str, fields: dict, timeout: int = 180) -> Any:
    # requests의 files 형식을 사용하면 multipart/form-data boundary가 자동 생성된다.
    multipart = {
        key: (None, str(value))
        for key, value in fields.items()
    }

    response = requests.post(
        cf_url(model),
        headers=auth_header(),
        files=multipart,
        timeout=timeout,
    )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Cloudflare 이미지 응답을 읽지 못했습니다. HTTP {response.status_code}"
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
        "properties": {key: {"type": "string"} for key in FIELDS},
        "required": FIELDS,
        "additionalProperties": False,
    }

    system_prompt = """
너는 대한민국 실종 재난문자의 인상착의 사실 추출기다.

절대 원칙:
- 원문에 실제로 있는 정보만 사용한다.
- 없는 정보는 빈 문자열("")로 둔다.
- 모호한 정보를 추측하지 않는다.
- 이름만 보고 국적, 피부톤, 머리 특징을 추측하지 않는다.

한국어 필드 작성 규칙:
1. "검은색 모자" -> hat_type="모자(종류 불명)", hat_color="검은색"
2. "회색 캡모자" -> hat_type="캡모자", hat_color="회색"
3. "검은색 바지" -> bottom="검은색 바지". 긴바지/반바지를 추측하지 않는다.
4. 피부톤, 곱슬/직모, 수염, 안경 등은 명시된 경우만 적는다.
5. ambiguity_notes에는 구체적으로 정할 수 없는 부분을 한국어로 적는다.

image_prompt_en 규칙:
- 반드시 자연스럽고 정확한 영어로 작성한다.
- 원문에 있는 사실만 포함한다.
- 현대의 일상복 기준으로 표현한다.
- 실제 얼굴 생김새를 창작하지 않는다.
- 예: "68-year-old Korean man, dark skin tone, short black curly hair,
  wearing a gray baseball cap, red short-sleeve T-shirt,
  black long pants, and black Crocs."

verification_requirements_en 규칙:
- 이미지에서 반드시 확인해야 할 명시된 인상착의만 영어로 적는다.
- 세미콜론(;)으로 구분한다.
- 예: "gray baseball cap; red short-sleeve T-shirt;
  black long pants; black Crocs; short black curly hair"
- 원문에 없는 특징은 절대로 추가하지 않는다.
""".strip()

    result = cloudflare_json_request(
        TEXT_MODEL,
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "다음 실종 재난문자를 정확히 분석해 주세요.\n\n"
                        f"{message}"
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 1400,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": schema,
            },
        },
    )

    parsed = result.get("response", result) if isinstance(result, dict) else result

    if isinstance(parsed, str):
        parsed = json.loads(parsed)

    if not isinstance(parsed, dict):
        raise RuntimeError("AI 분석 결과가 올바른 형식이 아닙니다.")

    return {
        key: str(parsed.get(key, "") or "").strip()
        for key in FIELDS
    }


# =========================================================
# 기본 인물 설정 / 상세도 검사
# =========================================================

def get_origin(features: dict) -> tuple[str, str]:
    nationality = features.get("nationality", "").strip()
    if nationality:
        return nationality, nationality
    return "한국인", "Korean"


def get_known_appearance_count(features: dict) -> int:
    return sum(
        1
        for key in APPEARANCE_FIELDS
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
# 이미지 생성
# =========================================================

def build_generation_prompt(
    features: dict,
    origin_en: str,
    correction: str = "",
) -> str:

    description = features.get("image_prompt_en", "").strip()
    requirements = features.get("verification_requirements_en", "").strip()

    if not description:
        raise RuntimeError("이미지 생성용 영어 인상착의 설명을 만들지 못했습니다.")

    correction_block = ""
    if correction:
        correction_block = f"""
PREVIOUS IMAGE FAILED VERIFICATION.
Fix ONLY these problems:
{correction}

Do not change any requirement that was already correct.
""".strip()

    prompt = f"""
Create a PHOTOREALISTIC full-body reference photograph of exactly ONE {origin_en} person.

PERSON DESCRIPTION:
{description}

MANDATORY VISIBLE REQUIREMENTS:
{requirements}

STYLE AND COMPOSITION:
- realistic contemporary everyday person
- modern ordinary clothing exactly matching the description
- natural front-facing standing pose
- entire body visible from head to feet
- plain white or very light gray studio background
- realistic camera photograph
- neutral documentary/reference-photo appearance

STRICT PROHIBITIONS:
- NO traditional clothing unless explicitly stated
- NO hanbok
- NO kimono
- NO historical robes
- NO ceremonial clothing
- NO fantasy clothing
- NO wizard clothing
- NO conical fantasy hat
- NO costume reinterpretation
- NO anime
- NO cartoon
- NO illustration
- NO poster design
- NO calligraphy
- NO text
- NO Korean characters
- NO Chinese characters
- NO Japanese characters
- NO English text
- NO numbers
- NO logos
- NO signs
- NO watermark

ACCURACY RULES:
- Every stated clothing color must match exactly.
- Every stated clothing type must match exactly.
- If a baseball cap is specified, it must be a baseball cap.
- If black long pants are specified, they must be black long pants.
- If Crocs are specified, they must look like Crocs, not sneakers.
- If skin tone or hair texture is specified, reflect it accurately.
- Do not invent distinctive facial details that were not provided.

{correction_block}
""".strip()

    return prompt[:3500]


def generate_image(prompt: str) -> tuple[bytes, str]:
    # FLUX.2 Klein은 REST API에서 multipart/form-data 사용
    result = cloudflare_multipart_request(
        IMAGE_MODEL,
        {
            "prompt": prompt,
            "width": 768,
            "height": 1024,
            # 값이 높을수록 프롬프트를 더 강하게 따르도록 유도
            "guidance": 4.0,
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
        for key in ("answer", "response", "result", "text", "caption"):
            value = result.get(key)
            if isinstance(value, str):
                return value

    return json.dumps(result, ensure_ascii=False)


def moondream_query(image_b64: str, question: str) -> str:
    data_uri = f"data:image/jpeg;base64,{image_b64}"

    result = cloudflare_json_request(
        VISION_MODEL,
        {
            "task": "query",
            "image": data_uri,
            "question": question,
            "reasoning": False,
            "temperature": 0.0,
            "max_tokens": 900,
            "stream": False,
        },
    )

    return extract_text_from_result(result)


def parse_json_loose(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
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


def verify_image(image_b64: str, features: dict) -> dict:
    requirements = features.get("verification_requirements_en", "").strip()

    question = f"""
Carefully verify this generated full-body reference image.

EXPLICIT REQUIRED FACTS:
{requirements}

Only evaluate facts explicitly listed above.
Do not penalize unspecified face details.

Reject the image if:
- any required clothing color is wrong
- any required clothing type is wrong
- required shoes are wrong
- required hat type/color is wrong
- specified hair or skin characteristics are wrong
- the full body is not visible
- traditional, historical, ceremonial or fantasy clothing appears without being required
- any text, calligraphy, letters, numbers, logo, sign, poster or watermark appears
- the image is anime/cartoon/illustration instead of a realistic reference photograph

Return JSON ONLY:
{{
  "score": 0,
  "pass": false,
  "missing": [],
  "wrong": [],
  "has_text": false,
  "feedback_en": ""
}}

score must be an integer from 0 to 100.
feedback_en must tell the image generator exactly what to correct.
""".strip()

    raw = moondream_query(image_b64, question)
    parsed = parse_json_loose(raw)

    if not parsed:
        return {
            "score": 0,
            "pass": False,
            "missing": [],
            "wrong": ["검수 결과를 읽지 못함"],
            "has_text": False,
            "feedback_en": (
                "Regenerate as a photorealistic contemporary full-body reference photo. "
                "Follow every explicit requirement exactly and include absolutely no text."
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

    if has_text:
        wrong.append("이미지 안에 글자가 있음")

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
        "feedback_en": str(parsed.get("feedback_en", "") or "").strip(),
        "raw": raw,
    }



# =========================================================
# 익명 사용 통계 / Supabase
# =========================================================

def analytics_enabled() -> bool:
    return bool(
        get_secret("SUPABASE_URL")
        and (get_secret("SUPABASE_SECRET_KEY") or get_secret("SUPABASE_KEY"))
    )


def supabase_key() -> str:
    return get_secret("SUPABASE_SECRET_KEY") or get_secret("SUPABASE_KEY")


def get_anonymous_user_id() -> str:
    """
    브라우저 쿠키에 익명 UUID를 저장한다.
    이름, 재난문자 원문, IP 주소 등은 통계 DB에 저장하지 않는다.
    """
    if "findvision_uid" in st.session_state:
        return st.session_state["findvision_uid"]

    try:
        saved = cookie_controller.get("findvision_uid")
    except Exception:
        saved = None

    if saved:
        user_id = str(saved)
    else:
        user_id = str(uuid.uuid4())
        try:
            cookie_controller.set("findvision_uid", user_id)
        except Exception:
            pass

    st.session_state["findvision_uid"] = user_id
    return user_id


def supabase_headers() -> dict:
    key = supabase_key()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def log_analytics_event(
    user_id: str,
    event_type: str,
    verification_pass=None,
    verification_score=None,
    attempts=None,
) -> None:
    if not analytics_enabled():
        return

    url = get_secret("SUPABASE_URL").rstrip("/") + "/rest/v1/analytics_events"

    payload = {
        "user_id": user_id,
        "event_type": event_type,
        "verification_pass": verification_pass,
        "verification_score": verification_score,
        "attempts": attempts,
    }

    try:
        response = requests.post(
            url,
            headers={
                **supabase_headers(),
                "Prefer": "return=minimal",
            },
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
    except Exception:
        # 통계 DB 장애가 본 기능을 막지 않도록 무시한다.
        pass


def fetch_analytics_events(max_rows: int = 10000) -> list:
    if not analytics_enabled():
        return []

    url = get_secret("SUPABASE_URL").rstrip("/") + "/rest/v1/analytics_events"
    rows = []
    page_size = 1000
    start = 0

    while start < max_rows:
        end = min(start + page_size - 1, max_rows - 1)

        response = requests.get(
            url,
            headers={
                **supabase_headers(),
                "Range": f"{start}-{end}",
            },
            params={
                "select": (
                    "user_id,event_type,created_at,"
                    "verification_pass,verification_score,attempts"
                ),
                "order": "created_at.asc",
            },
            timeout=20,
        )
        response.raise_for_status()

        chunk = response.json()
        if not chunk:
            break

        rows.extend(chunk)

        if len(chunk) < page_size:
            break

        start += page_size

    return rows


def parse_created_at(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def calculate_analytics(rows: list) -> dict:
    generations = [
        row for row in rows
        if row.get("event_type") == "image_generated"
    ]

    now = datetime.now(timezone.utc)
    seven_days_ago = now - timedelta(days=7)

    all_users = set()
    weekly_users = set()
    usage_dates = {}
    weekly_usage_dates = {}

    for row in generations:
        user_id = row.get("user_id")
        created_at = parse_created_at(row.get("created_at", ""))

        if not user_id or not created_at:
            continue

        all_users.add(user_id)
        usage_dates.setdefault(user_id, set()).add(created_at.date())

        if created_at >= seven_days_ago:
            weekly_users.add(user_id)
            weekly_usage_dates.setdefault(user_id, set()).add(created_at.date())

    returning_users = {
        user_id
        for user_id, dates in usage_dates.items()
        if len(dates) >= 2
    }

    weekly_returning_users = {
        user_id
        for user_id, dates in weekly_usage_dates.items()
        if len(dates) >= 2
    }

    verification_rows = [
        row for row in generations
        if row.get("verification_pass") is not None
    ]

    passed = sum(
        1 for row in verification_rows
        if row.get("verification_pass") is True
    )

    pass_rate = (
        100.0 * passed / len(verification_rows)
        if verification_rows else 0.0
    )

    attempts_values = [
        int(row["attempts"])
        for row in generations
        if row.get("attempts") is not None
    ]

    avg_attempts = (
        sum(attempts_values) / len(attempts_values)
        if attempts_values else 0.0
    )

    return {
        "total_users": len(all_users),
        "weekly_active_users": len(weekly_users),
        "returning_users": len(returning_users),
        "weekly_returning_users": len(weekly_returning_users),
        "total_generations": len(generations),
        "verification_pass_rate": pass_rate,
        "avg_attempts": avg_attempts,
    }


def show_admin_analytics() -> None:
    st.subheader("📊 ClueSight 서비스 사용 지표")

    if not analytics_enabled():
        st.info(
            "Supabase 통계 DB를 연결하면 사용 지표가 여기에 표시됩니다."
        )
        return

    admin_password = get_secret("ADMIN_PASSWORD")
    if not admin_password:
        st.warning("ADMIN_PASSWORD가 설정되어 있지 않습니다.")
        return

    entered = st.text_input(
        "관리자 비밀번호",
        type="password",
        key="analytics_admin_password",
    )

    if entered != admin_password:
        if entered:
            st.error("비밀번호가 맞지 않습니다.")
        return

    try:
        rows = fetch_analytics_events()
        metrics = calculate_analytics(rows)
    except Exception as exc:
        st.error(f"통계를 불러오지 못했습니다: {exc}")
        return

    a, b, c = st.columns(3)
    a.metric("누적 실제 사용자", f"{metrics['total_users']}명")
    b.metric(
        "최근 7일 실제 사용자",
        f"{metrics['weekly_active_users']}명",
        help="최근 7일 안에 이미지 생성을 1회 이상 완료한 사람",
    )
    c.metric(
        "재방문 사용자",
        f"{metrics['returning_users']}명",
        help="서로 다른 날짜에 이미지 생성을 2회 이상 완료한 사람",
    )

    d, e, f = st.columns(3)
    d.metric(
        "최근 7일 재방문 사용자",
        f"{metrics['weekly_returning_users']}명",
        help="최근 7일 동안 서로 다른 날짜에 2회 이상 사용한 사람",
    )
    e.metric("총 이미지 생성", f"{metrics['total_generations']}회")
    f.metric(
        "자동 검수 통과율",
        f"{metrics['verification_pass_rate']:.1f}%",
    )

    st.caption(
        f"평균 이미지 생성 시도 횟수: {metrics['avg_attempts']:.2f}회"
    )
    st.caption(
        "재난문자 원문과 개인정보는 통계 DB에 저장하지 않습니다."
    )


anonymous_user_id = get_anonymous_user_id()

if analytics_enabled() and not st.session_state.get("visit_logged", False):
    log_analytics_event(anonymous_user_id, "visit")
    st.session_state["visit_logged"] = True


# =========================================================
# UI
# =========================================================

st.title("🔎 FindVision AI")

st.caption(
    "상세 실종 재난문자를 AI가 분석하고, 인상착의를 반영한 "
    "현대적인 전신 참고 이미지를 생성한 뒤 Vision AI가 다시 검수합니다."
)

st.warning(
    "생성 이미지는 실제 실종자의 얼굴을 복원한 사진이 아닙니다. "
    "재난문자에 적힌 인상착의를 이해하기 위한 참고 자료입니다."
)


with st.expander("🔒 팀 관리자용 사용 통계", expanded=False):
    show_admin_analytics()

with st.expander("📌 권장 상세 재난문자 기준", expanded=True):
    st.markdown(
        """
가능하면 다음 정보를 포함해 주세요.

- 성별 / 나이 / 키 / 몸무게
- 피부톤
- 머리색 / 머리 길이 / 곱슬·직모 등 머리 형태
- 상의 색상과 종류
- 하의 색상과 종류
- 신발 색상과 종류
- 모자 색상과 정확한 종류
- 안경 / 수염 / 소지품 등 기타 특징

**없는 정보는 AI가 임의로 사실처럼 확정하지 않습니다.**
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
)

if st.button(
    "AI 분석 및 참고 이미지 생성",
    type="primary",
    use_container_width=True,
):

    if not message.strip():
        st.warning("실종 재난문자를 입력해 주세요.")
        st.stop()

    try:
        with st.spinner("재난문자의 인상착의를 분석하고 있습니다..."):
            features = extract_features(message.strip())

        origin_kr, origin_en = get_origin(features)

        st.subheader("1. AI가 분석한 인상착의")
        st.write(f"**기본 인물 설정:** {origin_kr}")

        display_keys = [
            "name", "gender", "age", "height", "weight",
            "skin_tone", "hair_color", "hair_length", "hair_texture",
            "top", "bottom", "shoes",
            "hat_type", "hat_color",
            "glasses", "facial_hair",
            "accessories", "special_features",
        ]

        c1, c2 = st.columns(2)

        for i, key in enumerate(display_keys):
            target = c1 if i < 9 else c2
            value = features.get(key, "").strip() or "정보 없음"
            target.write(f"**{LABELS[key]}:** {value}")

        ambiguity = features.get("ambiguity_notes", "").strip()
        if ambiguity:
            st.info(
                "⚠️ **AI가 임의로 추측하지 않은 모호한 정보:** "
                + ambiguity
            )

        known_count = get_known_appearance_count(features)
        missing = get_missing_recommended(features)

        if known_count < 3:
            st.error(
                "현재 재난문자에는 이미지를 안정적으로 생성하기 위한 "
                "인상착의 정보가 너무 적습니다."
            )
            if missing:
                st.write("**추가하면 좋은 정보:** " + ", ".join(missing))
            st.stop()

        if missing:
            st.warning(
                "일부 권장 정보가 없습니다: "
                + ", ".join(missing)
                + ". 없는 정보는 AI가 사실처럼 확정하지 않습니다."
            )

        with st.expander("🔧 이미지 생성 AI에 실제로 전달되는 영어 설명 확인"):
            st.write(features.get("image_prompt_en", ""))
            st.write("**필수 검수 조건:**")
            st.write(features.get("verification_requirements_en", ""))

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
                f"{attempt}차 이미지가 입력 인상착의와 맞는지 검수 중..."
            )

            verification = verify_image(image_b64, features)

            candidate = {
                "attempt": attempt,
                "image": image_bytes,
                "verification": verification,
            }

            if (
                best is None
                or verification["score"] > best["verification"]["score"]
            ):
                best = candidate

            if verification["pass"]:
                best = candidate
                break

            correction = verification["feedback_en"] or (
                "Generate a new photorealistic contemporary reference photograph. "
                "Correct every failed explicit appearance requirement. "
                "No text, calligraphy, traditional clothing, costume, or illustration."
            )

            progress.progress(
                int((attempt / MAX_ATTEMPTS) * 100)
            )

        progress.progress(100)
        status.empty()

        if best is None:
            raise RuntimeError("최종 이미지를 생성하지 못했습니다.")

        verdict = best["verification"]

        # 최종 이미지가 실제로 사용자에게 제공된 시점에 '실제 사용'으로 기록한다.
        log_analytics_event(
            anonymous_user_id,
            "image_generated",
            verification_pass=bool(verdict["pass"]),
            verification_score=int(verdict["score"]),
            attempts=int(best["attempt"]),
        )

        st.image(
            best["image"],
            caption=(
                f"{best['attempt']}차 생성 결과 · "
                f"자동 검수 점수 {verdict['score']}/100"
            ),
            use_container_width=True,
        )

        if verdict["pass"]:
            st.success("자동 검수를 통과한 이미지입니다.")
        else:
            st.warning(
                "3회 안에 모든 조건을 통과하지 못해 가장 높은 점수의 "
                "이미지를 표시했습니다. 최종 사용 전 사람이 확인해야 합니다."
            )

        if verdict["missing"]:
            st.write(
                "**누락된 항목:** "
                + ", ".join(map(str, verdict["missing"]))
            )

        if verdict["wrong"]:
            st.write(
                "**잘못 표현된 항목:** "
                + ", ".join(map(str, verdict["wrong"]))
            )

        with st.expander("원문 재난문자 보기"):
            st.write(message)

    except Exception as exc:
        st.error(f"오류가 발생했습니다: {exc}")
