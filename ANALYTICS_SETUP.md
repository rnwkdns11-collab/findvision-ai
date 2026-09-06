# FindVision v8 통계 기능 설정

추가 지표
- 누적 실제 사용자
- 최근 7일 실제 사용자(WAU)
- 재방문 사용자
- 최근 7일 재방문 사용자
- 총 이미지 생성 횟수
- 자동 검수 통과율
- 평균 생성 시도 횟수

실제 사용자 = 최종 이미지 생성까지 완료한 익명 사용자

## 1. Supabase
Supabase 프로젝트 생성 후 SQL Editor에서 `supabase_setup.sql` 전체 실행.

## 2. Streamlit Secrets
기존 Cloudflare 값은 그대로 두고 아래 3개를 추가.

```toml
SUPABASE_URL = "https://프로젝트ID.supabase.co"
SUPABASE_SECRET_KEY = "sb_secret_..."
ADMIN_PASSWORD = "원하는 관리자 비밀번호"
```

## 개인정보
통계에는 재난문자 원문, 이름, 피부톤, 복장 정보, IP 주소를 저장하지 않습니다.
브라우저 쿠키의 익명 UUID, 이벤트 시간, 이미지 생성/검수 결과만 저장합니다.
